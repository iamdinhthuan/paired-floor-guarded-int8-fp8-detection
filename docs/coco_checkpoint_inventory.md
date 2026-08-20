# COCO checkpoint inventory

The COCO arm uses official Ultralytics pretrained YOLO11 checkpoints and does not retrain them on COCO. The authoritative copies remain on the RTX 5090 host (`thuan@100.111.139.103`) under `/home/thuan/topic_c_ivc`.

| Model | Remote checkpoint | Bytes | SHA-256 |
|---|---|---:|---|
| YOLO11n | `/home/thuan/topic_c_ivc/yolo11n.pt` | 5,613,764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| YOLO11m | `/home/thuan/topic_c_ivc/yolo11m.pt` | 40,684,120 | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` |
| YOLO11x | `/home/thuan/topic_c_ivc/yolo11x.pt` | 114,636,239 | `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` |

The provenance records are:

- `manifests/training/coco_yolo11n_train_v1.json`
- `manifests/training/coco_yolo11m_train_v1.json`
- `manifests/training/coco_yolo11x_train_v1.json`

Download examples:

```bash
scp thuan@100.111.139.103:/home/thuan/topic_c_ivc/yolo11n.pt .
scp thuan@100.111.139.103:/home/thuan/topic_c_ivc/yolo11m.pt .
scp thuan@100.111.139.103:/home/thuan/topic_c_ivc/yolo11x.pt .
```

Verify after download:

```bash
sha256sum yolo11n.pt yolo11m.pt yolo11x.pt
```
