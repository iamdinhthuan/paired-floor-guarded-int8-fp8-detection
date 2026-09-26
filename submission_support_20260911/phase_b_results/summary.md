# Phase-B paired protocol — point-estimate summary

All values are AP points. Clean basis: original clean and
codec-control Q95. Corrupted mean = equal weight over the 12
corruption-severity cells. Retention = corrupted / Q95-clean.

## kitti / yolo11n  (kitti_val)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 64.7874 | 64.8360 | 44.7555 | 0.6903 | 14 |
| fp8 | 63.6778 | 63.8251 | 44.0638 | 0.6904 | 14 |
| int8 | 61.5173 | 61.6876 | 41.9334 | 0.6798 | 14 |

- fp8_minus_int8: G0=+2.1375  Gc=+2.1304  ΔE=-0.0071  (B=2000 paired ΔE 95% CI [-0.6734, +0.9087], basis=q95)
- fp32_minus_int8: G0=+3.1484  Gc=+2.8221  ΔE=-0.3263  (B=2000 paired ΔE 95% CI [-0.8272, +0.3977], basis=q95)

## kitti / yolo11m  (kitti_val)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 73.8201 | 73.8695 | 50.5849 | 0.6848 | 14 |
| fp8 | 73.3246 | 73.3084 | 50.3838 | 0.6873 | 14 |
| int8 | 69.2181 | 69.3417 | 47.0968 | 0.6792 | 14 |

- fp8_minus_int8: G0=+3.9667  Gc=+3.2870  ΔE=-0.6797  (B=2000 paired ΔE 95% CI [-1.3157, -0.0530], basis=q95)
- fp32_minus_int8: G0=+4.5278  Gc=+3.4881  ΔE=-1.0397  (B=2000 paired ΔE 95% CI [-1.5962, -0.5430], basis=q95)

## kitti / yolo11x  (kitti_val)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 72.9316 | 73.0582 | 51.8066 | 0.7091 | 14 |
| fp8 | 72.1968 | 72.2578 | 51.0454 | 0.7064 | 14 |
| int8 | 70.8741 | 70.8215 | 50.3040 | 0.7103 | 14 |

- fp8_minus_int8: G0=+1.4363  Gc=+0.7414  ΔE=-0.6949  (B=2000 paired ΔE 95% CI [-1.1322, -0.3292], basis=q95)
- fp32_minus_int8: G0=+2.2367  Gc=+1.5026  ΔE=-0.7341  (B=2000 paired ΔE 95% CI [-1.1545, -0.3790], basis=q95)

## voc / yolo11n  (voc_val)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 62.5433 | 62.5567 | 44.9581 | 0.7187 | 14 |
| fp8 | 62.0637 | 62.1889 | 44.7541 | 0.7196 | 14 |
| int8 | 61.4207 | 61.5109 | 43.7673 | 0.7115 | 14 |

- fp8_minus_int8: G0=+0.6780  Gc=+0.9868  ΔE=+0.3088  (B=2000 paired ΔE 95% CI [+0.0688, +0.5865], basis=q95)
- fp32_minus_int8: G0=+1.0458  Gc=+1.1908  ΔE=+0.1450  (B=2000 paired ΔE 95% CI [-0.0634, +0.3920], basis=q95)

## voc / yolo11m  (voc_val)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 68.5254 | 68.5612 | 52.0484 | 0.7592 | 14 |
| fp8 | 68.3122 | 68.3558 | 51.7964 | 0.7577 | 14 |
| int8 | 66.7483 | 66.6589 | 50.6744 | 0.7602 | 14 |

- fp8_minus_int8: G0=+1.6969  Gc=+1.1220  ΔE=-0.5749  (B=2000 paired ΔE 95% CI [-0.8031, -0.3428], basis=q95)
- fp32_minus_int8: G0=+1.9023  Gc=+1.3740  ΔE=-0.5283  (B=2000 paired ΔE 95% CI [-0.7501, -0.3037], basis=q95)

## voc / yolo11x  (voc_val)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 70.0865 | 70.0543 | 54.5573 | 0.7788 | 14 |
| fp8 | 70.0007 | 69.9407 | 54.3486 | 0.7771 | 14 |
| int8 | 69.0747 | 69.0275 | 53.2976 | 0.7721 | 14 |

- fp8_minus_int8: G0=+0.9132  Gc=+1.0510  ΔE=+0.1378  (B=2000 paired ΔE 95% CI [-0.0659, +0.3564], basis=q95)
- fp32_minus_int8: G0=+1.0268  Gc=+1.2597  ΔE=+0.2329  (B=2000 paired ΔE 95% CI [+0.0463, +0.4480], basis=q95)

## coco / yolo11n  (coco-subset-2000)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 39.4019 | 39.3243 | 28.9868 | 0.7371 | 14 |
| fp8 | 38.3540 | 38.5894 | 28.1891 | 0.7305 | 14 |
| int8 | 37.5989 | 37.4898 | 27.4133 | 0.7312 | 14 |

- fp8_minus_int8: G0=+1.0996  Gc=+0.7758  ΔE=-0.3238  (B=2000 paired ΔE 95% CI [-0.6682, +0.0119], basis=q95)
- fp32_minus_int8: G0=+1.8345  Gc=+1.5735  ΔE=-0.2610  (B=2000 paired ΔE 95% CI [-0.5317, +0.0531], basis=q95)

## coco / yolo11m  (coco-subset-2000)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 50.7085 | 50.4883 | 38.7820 | 0.7681 | 14 |
| fp8 | 50.1256 | 49.8628 | 38.0366 | 0.7628 | 14 |
| int8 | 48.8852 | 48.9741 | 36.5429 | 0.7462 | 14 |

- fp8_minus_int8: G0=+0.8887  Gc=+1.4937  ΔE=+0.6050  (B=2000 paired ΔE 95% CI [+0.2725, +0.9742], basis=q95)
- fp32_minus_int8: G0=+1.5142  Gc=+2.2391  ΔE=+0.7249  (B=2000 paired ΔE 95% CI [+0.4192, +1.0549], basis=q95)

## coco / yolo11x  (coco-subset-2000)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 54.5209 | 54.2422 | 42.6987 | 0.7872 | 14 |
| fp8 | 54.0701 | 53.9985 | 42.3760 | 0.7848 | 14 |
| int8 | 53.2097 | 52.9997 | 41.9022 | 0.7906 | 14 |

- fp8_minus_int8: G0=+0.9988  Gc=+0.4738  ΔE=-0.5250  (B=2000 paired ΔE 95% CI [-0.9251, -0.1832], basis=q95)
- fp32_minus_int8: G0=+1.2425  Gc=+0.7965  ΔE=-0.4460  (B=2000 paired ΔE 95% CI [-0.8064, -0.1525], basis=q95)

## kitti / retinanet_r50_fpn_v2  (kitti_val)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 50.1456 | - | 37.8041 | - | 13 |
| fp8-legacy | 49.1854 | - | 36.4790 | - | 13 |
| int8-legacy | 7.2143 | - | 7.4316 | - | 13 |
| fp8-matched512 | 49.4785 | 49.2521 | 36.7488 | 0.7461 | 14 |
| int8-matched512 | 28.2930 | 28.1827 | 20.3712 | 0.7228 | 14 |
| int8-selective512 | 49.6001 | 49.5304 | 36.5815 | 0.7386 | 14 |

- fp8matched_minus_int8matched: G0=+21.0694  Gc=+16.3776  ΔE=-4.6918  (B=2000 paired ΔE 95% CI [-5.4867, -3.8620], basis=q95)
- selective_minus_full_int8: G0=+21.3477  Gc=+16.2103  ΔE=-5.1374  (B=2000 paired ΔE 95% CI [-5.9409, -4.3360], basis=q95)

## voc / retinanet_r50_fpn_v2  (voc_val)

| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |
|---|---:|---:|---:|---:|---:|
| fp32 | 57.2341 | - | 42.3161 | - | 13 |
| fp8-legacy | 56.1698 | - | 41.4437 | - | 13 |
| int8-legacy | 33.6050 | - | 24.3229 | - | 13 |
| fp8-matched512 | 56.1722 | 56.2132 | 41.4776 | 0.7379 | 14 |
| int8-matched512 | 53.3037 | 53.1804 | 38.7063 | 0.7278 | 14 |
| int8-selective512 | 55.8692 | 55.7948 | 40.5205 | 0.7262 | 14 |

- fp8matched_minus_int8matched: G0=+3.0328  Gc=+2.7713  ΔE=-0.2615  (B=2000 paired ΔE 95% CI [-0.5780, +0.0607], basis=q95)
- selective_minus_full_int8: G0=+2.6144  Gc=+1.8142  ΔE=-0.8002  (B=2000 paired ΔE 95% CI [-0.9767, -0.6132], basis=q95)
