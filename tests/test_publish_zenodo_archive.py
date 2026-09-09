"""Offline fail-closed tests; no network requests or publication."""
import copy
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    'publish_archive', Path(__file__).parents[1] / 'scripts/publish_zenodo_archive.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def plan(tmp_path):
    payload = b'verified scientific archive'
    (tmp_path / 'example.zip').write_bytes(payload)
    item = {'md5': hashlib.md5(payload).hexdigest(),
            'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    return {'deposition_id': 22664869, 'doi': '10.5281/zenodo.22664869',
            'conceptrecid': '22031663', 'version': '2.2.0',
            'metadata': {'version': '2.2.0', 'title': 'Approved title'},
            'files': {'example.zip': item}}


def draft(plan, uploaded=True):
    return {'id': 22664869, 'conceptrecid': '22031663', 'state': 'unsubmitted',
            'submitted': False, 'metadata': copy.deepcopy(plan['metadata']),
            'files': [{'filename': name, 'filesize': item['bytes'], 'checksum': item['md5']}
                      for name, item in plan['files'].items()] if uploaded else [],
            'links': {'bucket': 'https://zenodo.org/api/files/test-bucket'}}


def test_matching_archive(plan, tmp_path):
    m.verify_files(draft(plan)['files'], plan['files'])
    m.verify_local(tmp_path / 'example.zip', plan['files']['example.zip'])


@pytest.mark.parametrize('change', ['missing', 'extra', 'checksum', 'size', 'duplicate'])
def test_bad_file_set_rejected(plan, change):
    files = draft(plan)['files']
    if change == 'missing':
        files.clear()
    elif change == 'extra':
        files[0]['filename'] = 'unexpected.zip'
    elif change == 'duplicate':
        files.append(copy.deepcopy(files[0]))
    else:
        files[0]['checksum' if change == 'checksum' else 'filesize'] = '0'
    with pytest.raises(m.ContractError):
        m.verify_files(files, plan['files'])


def test_local_sha_failure(plan, tmp_path):
    (tmp_path / 'example.zip').write_bytes(b'x' * plan['files']['example.zip']['bytes'])
    with pytest.raises(m.ContractError, match='SHA-256'):
        m.verify_local(tmp_path / 'example.zip', plan['files']['example.zip'])


@pytest.mark.parametrize('change', ['id', 'conceptrecid', 'metadata'])
def test_modified_draft_rejected(plan, change):
    data = draft(plan)
    if change == 'metadata':
        data['metadata']['title'] = 'Unapproved edit'
    else:
        data[change] = 123
    with pytest.raises(m.ContractError):
        m.verify_metadata(data, plan)


def test_matching_uploaded_files_not_reuploaded(plan, tmp_path, monkeypatch):
    api = Mock(return_value=draft(plan))
    monkeypatch.setattr(m, 'zenodo', api)
    m.resume_once(plan, tmp_path, 'test-secret', publish=False)
    assert [call.args[0] for call in api.call_args_list] == ['GET', 'GET']


def test_metadata_changed_before_publish_stops(plan, tmp_path, monkeypatch):
    changed = draft(plan)
    changed['metadata']['title'] = 'Changed during upload'
    api = Mock(side_effect=[draft(plan), changed])
    monkeypatch.setattr(m, 'zenodo', api)
    with pytest.raises(m.ContractError, match='Metadata changed'):
        m.resume_once(plan, tmp_path, 'test-secret', publish=True)
    assert not any(call.args[0] == 'POST' for call in api.call_args_list)


def test_publish_after_verified_inventory_only(plan, tmp_path, monkeypatch):
    api = Mock(side_effect=[draft(plan), draft(plan), {}])
    monkeypatch.setattr(m, 'zenodo', api)
    public = Mock(return_value={'doi': plan['doi']})
    monkeypatch.setattr(m, 'verify_public', public)
    assert m.resume_once(plan, tmp_path, 'test-secret', True)['doi'] == plan['doi']
    assert [call.args[0] for call in api.call_args_list] == ['GET', 'GET', 'POST']
    assert api.call_args.args[1].endswith('/22664869/actions/publish')
    public.assert_called_once_with(plan)


def test_already_published_does_not_republish(plan, tmp_path, monkeypatch):
    data = draft(plan)
    data['submitted'] = True
    api = Mock(return_value=data)
    monkeypatch.setattr(m, 'zenodo', api)
    monkeypatch.setattr(m, 'verify_public', Mock(return_value={'doi': plan['doi']}))
    m.resume_once(plan, tmp_path, 'test-secret', True)
    assert api.call_count == 1


def test_credentials_never_sent_to_other_host(monkeypatch):
    api = Mock()
    monkeypatch.setattr(m.requests, 'request', api)
    with pytest.raises(m.ContractError):
        m.request('GET', 'https://not-zenodo.example/api', 'test-secret')
    api.assert_not_called()


def test_504_is_retryable_without_response_body(monkeypatch):
    monkeypatch.setattr(m.requests, 'request', Mock(return_value=Mock(status_code=504)))
    with pytest.raises(m.TransientError, match='^HTTP 504$'):
        m.request('GET', 'https://zenodo.org/api/records/1', 'test-secret')
