#!/usr/bin/env python3
"""Layerwise conv-MAC share of the torchvision RetinaNet regression head.

Supports the practical-guidance statement that the regression-head subgraph is
"near a quarter" of convolution MACs at the frozen input size. Counts every
Conv2d firing via forward hooks (the heads fire once per FPN level, which is
why the share is this large). MACs per firing =
in_channels * k_h * k_w * out_channels * out_h * out_w.

torchvision 0.21, retinanet_resnet50_fpn_v2, eval mode.
"""
import torch
import torchvision


def mac_share(num_classes: int) -> dict:
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=num_classes).eval()
    macs = {}

    def make_hook(name):
        def hook(mod, _inp, out):
            macs[name] = macs.get(name, 0) + (
                mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
                * mod.out_channels * out.shape[2] * out.shape[3])
        return hook

    handles = [mod.register_forward_hook(make_hook(n))
               for n, mod in model.named_modules()
               if isinstance(mod, torch.nn.Conv2d)]
    with torch.no_grad():
        # engines see the frozen 640x640 letterbox directly (no torchvision
        # transform), so feed the backbone and then the heads manually
        feats = model.backbone(torch.zeros(1, 3, 640, 640))
        model.head(list(feats.values()))
    for h in handles:
        h.remove()

    groups = {"regression_head": 0, "classification_head": 0,
              "fpn": 0, "backbone": 0}
    for name, m in macs.items():
        if "head.regression_head" in name:
            groups["regression_head"] += m
        elif "head.classification_head" in name:
            groups["classification_head"] += m
        elif "backbone.fpn" in name:
            groups["fpn"] += m
        else:
            groups["backbone"] += m
    total = sum(groups.values())
    shares = {k: 100 * v / total for k, v in groups.items()}
    shares["_total_GMAC"] = total / 1e9
    return shares


if __name__ == "__main__":
    # model classes per the training manifests: 9 = KITTI (8 fg + bg),
    # 21 = VOC (20 fg + bg)
    for nc in (9, 21):
        shares = mac_share(nc)
        print(f"num_classes={nc}: " +
              ", ".join(f"{k}={v:.1f}%" for k, v in shares.items()
                        if not k.startswith("_")) +
              f" | total={shares['_total_GMAC']:.1f} GMAC")
