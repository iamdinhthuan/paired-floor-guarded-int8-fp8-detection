# Multi-seed compact evidence

This directory records the balanced YOLO11m severity-5 sensitivity analysis
over VOC, KITTI, and TT100K. The grid contains three training seeds, three
calibration seeds, two formats, one JPEG-95 matched-clean control, and four
corruptions. The 270 metrics produce 108 direct format-contrast cells.

`analysis_complete.json` binds the compact analysis artifacts and the metric
completion report. `metric_complete.json` binds all 270 metric records and the
inference completion report. The latter records the remote prediction/input/run
artifact hashes; those 1.5 GiB prediction payloads are intentionally not copied
into this editable Overleaf source package. This is therefore a compact audit
surface, not a replacement for the future public evidence archive.
