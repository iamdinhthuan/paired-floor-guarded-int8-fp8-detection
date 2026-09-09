"""Resume one frozen Zenodo deposit; never create, edit metadata, or delete files.

Only a complete, hash-matched archive with unchanged approved metadata may be
published. GitHub draft assets are transport storage, not a public release.
Run without --publish to upload/verify only. Uses environment-held credentials.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote, urlparse

import requests


class TransientError(RuntimeError):
    """An outage/rate limit permits a bounded retry after re-reading state."""


class ContractError(RuntimeError):
    """Changed state or integrity failure must stop, never auto-repair."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def request(method, url, token=None, host='zenodo.org', **kwargs):
    parsed = urlparse(url)
    require(parsed.scheme == 'https' and parsed.netloc == host and not parsed.username,
            'Unexpected API destination')
    headers = kwargs.pop('headers', {})
    if token:
        headers['Authorization'] = 'Bearer ' + token
    try:
        response = requests.request(method, url, headers=headers,
            timeout=(300, 300) if method == 'PUT' else (15, 45),
            allow_redirects=False, **kwargs)
    except requests.RequestException as error:
        raise TransientError(type(error).__name__) from None
    if response.status_code == 429 or response.status_code >= 500:
        raise TransientError('HTTP ' + str(response.status_code))
    require(response.ok or response.status_code in (302, 307),
            'API rejected request: HTTP ' + str(response.status_code) + ' at ' + parsed.path)
    return response


def zenodo(method, url, token=None, **kwargs):
    response = request(method, url, token, **kwargs)
    require(response.status_code < 300, 'Unexpected Zenodo redirect')
    return response.json()


def verify_local(path, item):
    require(path.is_file() and path.stat().st_size == item['bytes'],
            'Local size mismatch: ' + path.name)
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    require(digest.hexdigest() == item['sha256'], 'Local SHA-256 mismatch: ' + path.name)


def fetch_assets(plan, directory, github_token):
    base = 'https://api.github.com/repos/' + plan['repository']
    # Require the reviewed tag to keep identifying the frozen source commit.
    tag = request('GET', base + '/git/ref/tags/v2.2.0', github_token,
                  host='api.github.com').json()['object']
    if tag['type'] == 'tag':
        tag = request('GET', base + '/git/tags/' + tag['sha'], github_token,
                      host='api.github.com').json()['object']
    require(tag['type'] == 'commit' and tag['sha'] == plan['release_commit'],
            'Frozen release tag changed')
    directory.mkdir(parents=True, exist_ok=True)
    for name, item in plan['files'].items():
        require(Path(name).name == name and name not in ('.', '..'), 'Unsafe asset filename')
        path = directory / name
        if not path.exists():
            response = request('GET', base + '/releases/assets/' + str(item['asset_id']),
                github_token, host='api.github.com',
                headers={'Accept': 'application/octet-stream'}, stream=True)
            if response.status_code in (302, 307):
                location = response.headers['Location']
                parsed = urlparse(location)
                require(parsed.scheme == 'https' and parsed.hostname in
                        ('release-assets.githubusercontent.com', 'objects.githubusercontent.com'),
                        'Unexpected asset redirect')
                response.close()
                # The signed blob URL receives NO GitHub or Zenodo credentials.
                response = request('GET', location, host=parsed.netloc, stream=True)
            require(response.status_code == 200, 'Asset response was not binary success')
            temporary = path.with_name(path.name + '.part')
            try:
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
                verify_local(temporary, item)
                temporary.rename(path)
            except requests.RequestException as error:
                raise TransientError(type(error).__name__) from None
            finally:
                response.close()
        verify_local(path, item)
        print('Local SHA-256 verified: ' + name, flush=True)


def verify_files(records, expected, *, public=False, complete=True):
    key = 'key' if public else 'filename'
    names = [record[key] for record in records]
    require(len(names) == len(set(names)), 'Duplicate remote filenames')
    require(set(names) <= set(expected), 'Unexpected remote file; no automatic deletion')
    if complete:
        require(set(names) == set(expected), 'Archive file set incomplete')
    for record in records:
        name = record[key]
        require(record['checksum'].removeprefix('md5:') == expected[name]['md5'],
                'Remote checksum mismatch: ' + name)
        size = record['size'] if public else record['filesize']
        require(int(size) == expected[name]['bytes'], 'Remote size mismatch: ' + name)


def verify_metadata(draft, plan):
    require(str(draft['id']) == str(plan['deposition_id']), 'Wrong deposition')
    require(str(draft['conceptrecid']) == plan['conceptrecid'], 'Wrong concept record')
    for field, expected in plan['metadata'].items():
        require(draft['metadata'].get(field) == expected, 'Metadata changed: ' + field)


def verify_public(plan):
    record = zenodo('GET', 'https://zenodo.org/api/records/' + str(plan['deposition_id']))
    require(record['doi'] == plan['doi'], 'Published DOI mismatch')
    require(record['metadata']['version'] == plan['version'], 'Published version mismatch')
    for field in ('title', 'creators', 'description', 'publication_date'):
        require(record['metadata'].get(field) == plan['metadata'][field],
                'Published metadata mismatch: ' + field)
    verify_files(record['files'], plan['files'], public=True)
    return record


def resume_once(plan, directory, token, publish):
    base = 'https://zenodo.org/api/deposit/depositions/' + str(plan['deposition_id'])
    draft = zenodo('GET', base, token)
    # A prior publish may have succeeded despite a client-side timeout.
    if draft['submitted']:
        return verify_public(plan)
    verify_metadata(draft, plan)
    require(draft['state'] == 'unsubmitted', 'Deposit is not an editable unpublished draft')
    verify_files(draft['files'], plan['files'], complete=False)
    existing = {item['filename'] for item in draft['files']}
    bucket = draft['links']['bucket']
    require(bucket.startswith('https://zenodo.org/api/files/'), 'Unexpected bucket route')
    for name, item in plan['files'].items():
        if name in existing:
            print('Already uploaded and verified: ' + name, flush=True)
            continue
        path = directory / name
        verify_local(path, item)
        print('Uploading: ' + name, flush=True)
        with path.open('rb') as payload:
            receipt = zenodo('PUT', bucket.rstrip('/') + '/' + quote(name, safe=''),
                token, data=payload, headers={'Content-Type': 'application/octet-stream'})
        require(receipt['checksum'].removeprefix('md5:') == item['md5'],
                'Upload receipt mismatch: ' + name)
    # Re-read everything, including metadata, immediately before publication.
    draft = zenodo('GET', base, token)
    verify_metadata(draft, plan)
    verify_files(draft['files'], plan['files'])
    require(not draft['submitted'], 'Draft changed concurrently; recheck public record')
    if not publish:
        print('Upload complete and verified; publication not requested.', flush=True)
        return None
    print('All ten files and approved metadata verified; publishing existing draft.', flush=True)
    zenodo('POST', base + '/actions/publish', token)
    return verify_public(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=Path('publication-assets'))
    parser.add_argument('--attempts', type=int, default=6)
    parser.add_argument('--retry-delay', type=int, default=300)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    require(1 <= args.attempts <= 12 and 0 <= args.retry_delay <= 600, 'Invalid retry bounds')
    plan = json.loads(args.plan.read_text())
    require(plan['deposition_id'] == 22664869 and plan['version'] == '2.2.0'
            and plan['doi'] == '10.5281/zenodo.22664869' and len(plan['files']) == 10,
            'Plan is not the authorized frozen publication')
    require(bool(os.environ['GH_TOKEN']), 'Missing GitHub token')
    if args.download_only:
        require(not args.publish, '--download-only cannot publish')
        fetch_assets(plan, args.directory, os.environ['GH_TOKEN'])
        return
    token = os.environ['ZENODO_PUBLICATION_TOKEN']
    require(bool(token), 'Missing Zenodo secret')
    for attempt in range(1, args.attempts + 1):
        print(f'Publication attempt {attempt}/{args.attempts}', flush=True)
        try:
            # Only download the large transport files once the authenticated API responds.
            zenodo('GET', 'https://zenodo.org/api/deposit/depositions/22664869', token)
            fetch_assets(plan, args.directory, os.environ['GH_TOKEN'])
            result = resume_once(plan, args.directory, token, args.publish)
            if result:
                (args.directory / 'public-record.json').write_text(json.dumps(result, indent=2))
                print('PUBLICATION VERIFIED: https://doi.org/' + plan['doi'], flush=True)
                summary = os.environ.get('GITHUB_STEP_SUMMARY')
                if summary:
                    with open(summary, 'a') as handle:
                        handle.write('Published and anonymously verified: https://doi.org/' + plan['doi'] + '\n')
            return
        except TransientError as error:
            print('Temporary service failure: ' + str(error), flush=True)
            if attempt == args.attempts:
                raise SystemExit('Retry budget exhausted; DOI publication NOT verified.')
            time.sleep(args.retry_delay)


if __name__ == '__main__':
    try:
        main()
    except ContractError as error:
        sys.exit('Publication stopped safely: ' + str(error))
