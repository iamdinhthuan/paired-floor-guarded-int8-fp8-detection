"""Manifest-aware COCO selection; official image IDs are never replaced."""
from __future__ import annotations

from pathlib import Path

from .manifest import ManifestError, official_coco_images, ordered_manifest_images


def load_coco_images(annotation_path: str, *, image_root: str | None = None, image_manifest: str | None = None, manifest_cache_root: str | None = None, limit: int = 0) -> list[tuple[int, str]]:
    if bool(image_root) == bool(image_manifest):
        raise ManifestError("provide exactly one of --image-root or --image-manifest")
    official = official_coco_images(annotation_path)
    if image_root:
        images = [(int(item["id"]), str((Path(image_root) / item["file_name"]).resolve())) for item in official]
    else:
        if not manifest_cache_root:
            raise ManifestError("--manifest-cache-root is required with --image-manifest")
        images = ordered_manifest_images(image_manifest, manifest_cache_root)
        official_ids = {int(item["id"]) for item in official}
        ids = [image_id for image_id, _ in images]
        if ids != sorted(ids) or any(image_id not in official_ids for image_id in ids):
            raise ManifestError("manifest IDs must be valid official COCO IDs in ascending order")
    if limit:
        images = images[:limit]
    if not images or len({image_id for image_id, _ in images}) != len(images):
        raise ManifestError("selected images are empty or contain duplicate IDs")
    for image_id, path in images:
        if not Path(path).is_file():
            raise ManifestError(f"missing bytes for COCO image_id {image_id}: {path}")
    return images


def letterbox(im, new_shape: int = 640, color: int = 114):
    import cv2
    h, w = im.shape[:2]
    gain = min(new_shape / h, new_shape / w)
    nh, nw = round(h * gain), round(w * gain)
    if (nw, nh) != (w, h):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = (new_shape - nw) / 2, (new_shape - nh) / 2
    top, bottom, left, right = round(ph - 0.1), round(ph + 0.1), round(pw - 0.1), round(pw + 0.1)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color)), gain, (left, top)


def preprocess(image_path: str, imgsz: int):
    import cv2
    import numpy as np
    im0 = cv2.imread(image_path)
    if im0 is None:
        raise ManifestError(f"OpenCV cannot read image: {image_path}")
    h0, w0 = im0.shape[:2]
    im, gain, (padx, pady) = letterbox(im0, imgsz)
    return np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0, gain, padx, pady, w0, h0
