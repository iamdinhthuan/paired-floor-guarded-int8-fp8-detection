# Frozen v2.2.0 publication

The manual **Zenodo publication** workflow resumes existing deposition
`22664869`, reserved DOI `10.5281/zenodo.22664869`. A reserved DOI is not proof
of publication. Success requires the final anonymous public-record check.

## Run

In Actions, choose **Zenodo publication → Run workflow** on `main`.
`probe` tests the public API without credentials. `publish` downloads the ten
reviewed files from an unpublished GitHub staging release, verifies SHA-256,
then resumes upload and publishes only after the complete Zenodo inventory,
MD5 checksums, sizes, and approved metadata have been verified again.

The default is six attempts, with five-minute backoff only for temporary
network errors, HTTP 429, or HTTP 5xx. Persistent failures stop the run; no
recurring schedule is installed. Rerunning resumes matching uploaded files.
Changed metadata, unexpected files, and checksum mismatches stop immediately
without deleting or overwriting them. Concurrent runs are serialized.

`ZENODO_PUBLICATION_TOKEN` is a repository Actions secret. The GitHub token is
the short-lived workflow token. The publication job grants it `contents: write`
because GitHub restricts unpublished release assets to callers with push
access; the client only reads those assets and never writes GitHub releases.
The credential-free probe retains `contents: read`.
Tokens are not stored in files, artifacts,
commits, or command-line arguments. The Zenodo secret is exposed only to the
publication step, not dependency installation or third-party actions. Revoke
or remove the repository secret after publication if it is no longer needed.

## Scope and recovery

Scientific source remains frozen at tag `v2.2.0`, commit
`c9899eace4725f9f7fc41af4afda58698235ad07`. Later automation commits do not
change the archived source ZIP, manuscript PDFs, or evidence checksums.
The workflow does not create versions, change authors/licenses, or run models.

The GitHub release named **Zenodo transfer staging v2.2.0 — DO NOT PUBLISH**
is deliberately a draft. Do not publish it: GitHub–Zenodo integration could
otherwise create a second deposit. Keep using the existing Zenodo record.
No unpublished staging asset should be described as publicly accessible.

If Zenodo is unavailable, the files remain staged and the DOI remains only
reserved unless the public-record check establishes publication. A timeout
after the publish request is ambiguous: the next run re-reads the deposition
and, if already submitted, verifies the public record without republishing.

The ten-file transport manifest is `zenodo-v2.2.0.json`. Any change to that
frozen manifest or metadata requires a new explicit publication review.
