import hashlib
import zipfile

import pytest

import acquire_coco_pilot
from acquire_coco_pilot import download, extract_verified, validate_relative


def test_extracts_only_named_member_and_checks_hash(tmp_path):
    archive = tmp_path / "test.zip"
    content = b"trusted-image"
    digest = hashlib.sha256(content).hexdigest()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("val2017/image.jpg", content)
        bundle.writestr("../untrusted", b"never-extracted")
    destination = tmp_path / "image.jpg"
    with zipfile.ZipFile(archive) as bundle:
        extract_verified(bundle, "val2017/image.jpg", destination, digest)
        assert destination.read_bytes() == content
        extract_verified(bundle, "val2017/image.jpg", destination, digest)
        with pytest.raises(ValueError, match="hash"):
            extract_verified(bundle, "val2017/image.jpg", destination, "0" * 64)
    assert not (tmp_path / "untrusted").exists()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "images/../escape", "images/train2017/not-a-coco-id.jpg"])
def test_rejects_noncanonical_calibration_paths(path):
    with pytest.raises(ValueError):
        validate_relative(path)


def test_accepts_frozen_calibration_image_path():
    assert validate_relative("images/train2017/000000000081.jpg") == "000000000081.jpg"


def test_download_skips_existing_verified_file_without_network(tmp_path):
    content = b"already-downloaded"
    destination = tmp_path / "image.jpg"
    destination.write_bytes(content)
    download("file:///definitely/not/a/real/path", destination, hashlib.sha256(content).hexdigest())
    assert destination.read_bytes() == content
    with pytest.raises(ValueError, match="hash"):
        download("file:///definitely/not/a/real/path", destination, "0" * 64)


def test_download_fetches_and_verifies(tmp_path):
    source = tmp_path / "source.jpg"
    content = b"payload"
    source.write_bytes(content)
    destination = tmp_path / "out" / "image.jpg"
    download(source.as_uri(), destination, hashlib.sha256(content).hexdigest())
    assert destination.read_bytes() == content
    with pytest.raises(ValueError, match="hash"):
        download(source.as_uri(), tmp_path / "bad.jpg", "0" * 64)
    assert not (tmp_path / "bad.jpg").exists()


def test_download_retries_transient_errors_and_cleans_partial(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    content = b"payload"
    source.write_bytes(content)
    calls = []
    real = acquire_coco_pilot.urllib.request.urlopen

    def flaky(url, *args, **kwargs):
        calls.append(url)
        if len(calls) <= 2:
            raise OSError("connection reset")
        return real(url, *args, **kwargs)

    monkeypatch.setattr(acquire_coco_pilot.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(acquire_coco_pilot.time, "sleep", lambda _: None)
    destination = tmp_path / "image.jpg"
    destination.with_suffix(".jpg.partial").write_bytes(b"stale")
    download(source.as_uri(), destination, hashlib.sha256(content).hexdigest())
    assert destination.read_bytes() == content
    assert len(calls) == 3
