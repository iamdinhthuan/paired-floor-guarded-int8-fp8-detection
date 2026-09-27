#!/usr/bin/env python3
"""Train one resumable FCOS-R50-FPN transfer checkpoint from YOLO labels.

Mirrors train_retinanet_dataset.py exactly; the only differences are the
torchvision model and that FCOS uses sigmoid classification with no
reserved background label (labels stay 0-based).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead

from topic_c.manifest import sha256_file
from train_retinanet_dataset import (
    YoloDetectionDataset,
    collate,
    mean_validation_loss,
    resolve_training_roots,
)


def build_model(num_classes: int, image_size: int):
    model = fcos_resnet50_fpn(
        weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
        min_size=image_size,
        max_size=image_size,
    )
    model.head.classification_head = FCOSClassificationHead(
        256, 1, num_classes, norm_layer=partial(nn.GroupNorm, 32)
    )
    return model


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset", choices=("voc", "kitti", "tt100k"), required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--acquisition-registry", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registry-out", required=True)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    root, profile_path = Path(args.project_root).resolve(), Path(args.profile).resolve()
    data_path, acquisition_path = Path(args.data_yaml).resolve(), Path(args.acquisition_registry).resolve()
    registry_out = Path(args.registry_out).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"]).resolve()
    image_size = int(profile["imgsz_by_dataset"][args.dataset])
    batch_size = int(profile["batch_by_dataset"][args.dataset])
    workers = int(profile["workers"])
    epochs = int(profile["epochs"])
    output = root / "outputs" / "training" / args.dataset / args.run_id
    weights_dir = output / "weights"
    resume = Path(args.resume_from).resolve() if args.resume_from else None
    if registry_out.exists():
        raise SystemExit("FCOS TRAINING REFUSED: registry already exists")
    if output.exists() and resume is None:
        raise SystemExit("FCOS TRAINING REFUSED: partial output requires --resume-from")
    if resume is not None and (resume != (weights_dir / "last.pt").resolve() or not resume.is_file()):
        raise SystemExit("FCOS TRAINING REFUSED: invalid resume checkpoint")
    marker = acquisition_path.with_suffix(acquisition_path.suffix + ".complete")
    if not acquisition_path.is_file() or not marker.is_file() or marker.read_text().strip() != sha256_file(acquisition_path):
        raise SystemExit("FCOS TRAINING REFUSED: acquisition registry is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    seed = int(profile["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(profile["deterministic"]), warn_only=True)
    names = data["names"]
    num_classes = len(names) if isinstance(names, (list, dict)) else 0
    train_roots = resolve_training_roots(dataset_root, data["train"])
    val_roots = resolve_training_roots(dataset_root, data["val"])
    # FCOS is a sigmoid detector without a reserved background label.
    train_set = YoloDetectionDataset(dataset_root, train_roots, augment=True, label_offset=0)
    val_set = YoloDetectionDataset(dataset_root, val_roots, augment=False, label_offset=0)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                              collate_fn=collate, pin_memory=True, generator=generator, persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                            collate_fn=collate, pin_memory=True, persistent_workers=workers > 0)
    device = torch.device("cuda:0")
    model = build_model(num_classes, image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(profile["lr"]), weight_decay=float(profile["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(profile["amp"]))
    start_epoch, best_loss = 0, float("inf")
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"]); start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])
    for epoch in range(start_epoch, epochs):
        model.train(); running = []
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(profile["amp"])):
                loss = sum(model(images, targets).values())
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                print(f"epoch={epoch + 1}/{epochs} step={step}/{len(train_loader)} loss={np.mean(running[-100:]):.6f}", flush=True)
        validation_loss = mean_validation_loss(model, val_loader, device)
        state = {"schema_version": 1, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "best_validation_loss": min(best_loss, validation_loss),
                 "profile_sha256": sha256_file(profile_path), "data_yaml_sha256": sha256_file(data_path)}
        atomic_save(state, weights_dir / "last.pt")
        if validation_loss < best_loss:
            best_loss = validation_loss
            atomic_save(
                {
                    "schema_version": 1,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "validation_loss": validation_loss,
                    "profile_sha256": sha256_file(profile_path),
                    "data_yaml_sha256": sha256_file(data_path),
                },
                weights_dir / "best.pt",
            )
        print(f"epoch={epoch + 1}/{epochs} train_loss={np.mean(running):.6f} validation_loss={validation_loss:.6f}", flush=True)
    record = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
              "dataset": args.dataset, "model": "fcos-r50-fpn", "epochs": epochs, "seed": seed,
              "profile": str(profile_path), "profile_sha256": sha256_file(profile_path), "data_yaml": str(data_path),
              "data_yaml_sha256": sha256_file(data_path), "acquisition_registry_sha256": sha256_file(acquisition_path),
              "best_weights": str(weights_dir / "best.pt"), "best_weights_sha256": sha256_file(weights_dir / "best.pt"),
              "selection": "minimum validation loss", "num_classes_excluding_background": num_classes}
    registry_out.parent.mkdir(parents=True, exist_ok=True)
    registry_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    registry_out.with_suffix(registry_out.suffix + ".complete").write_text(sha256_file(registry_out) + "\n", encoding="utf-8")
    (weights_dir / "last.pt").unlink(missing_ok=True)
    print(json.dumps({"TRAINING COMPLETE": args.run_id, "registry_sha256": sha256_file(registry_out)}, indent=2))


if __name__ == "__main__":
    main()
