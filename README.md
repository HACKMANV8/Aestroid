## Overview

This repo provides an end-to-end pipeline to:
- detect people in a drone video,
- map detections to GPS coordinates using a GPS log,
- merge with known threat positions, and
- run a strategic terrain analysis.

The entrypoint is `run_pipeline.py`.

## Prerequisites

- Python 3.10+
- Packages:
  - ultralytics, torch (GPU build optional), opencv-python, pandas, numpy, scipy, pygame (for visualization)
- Optional NVIDIA GPU + CUDA-enabled PyTorch for faster detection

Install example (CPU, minimal):
```bash
pip install ultralytics opencv-python pandas numpy scipy pygame
```

Install PyTorch with CUDA (example, CUDA 12.4 wheels):
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

## Required Files

Place these in the project root (`/home/bb/dataset/cdni43p_v3r1`):

- `footage.mp4` (input video)
- `drone_gps_log.csv` (frame-aligned GPS log)
  - Generate with:
    ```bash
    python create_gps_log.py --video footage.mp4 --skip-frames 1 --terrain terrain_data.jsonl --out drone_gps_log.csv
    ```
  - Tip: set `--skip-frames` to the same value used for detection.
- `terrain_data.jsonl` (terrain dataset with columns `longitude`/`latitude` and `elevation_m`)
- `threat_positions.csv` (known enemy positions; columns `longitude`, `latitude`)

Optional:
- Custom YOLO weights via `--model` (e.g., `ultralytics/yolov8l`)

## What the Pipeline Does

`run_pipeline.py` performs three steps:
1) Runs `detect_people.py` on `footage.mp4` to produce raw bounding boxes (`raw_detections.csv`).
2) Converts raw pixel detections to GPS using `drone_gps_log.csv`, writing `detected_persons.csv`.
3) Runs `ml.py` to analyze terrain and threats (base threats + detected persons), writing `Strategic_Analysis_Output.csv`.

If `threat_positions.csv` contains more than 2 rows, their coordinates are also appended to `detected_persons.csv` (deduplicated).

## Quick Start

Run the full pipeline for the first 30 seconds, with reasonable defaults:
```bash
python run_pipeline.py \
  --video footage.mp4 \
  --gps drone_gps_log.csv \
  --terrain terrain_data.jsonl \
  --threat-positions threat_positions.csv \
  --imgsz 960 \
  --min-person-confidence 0.5 \
  --max-seconds 30
```

View results in pygame (3D terrain viewer):
```bash
python pygame2.py
```

## Key Arguments (run_pipeline.py)

- `--video`: path to input video (default: `footage.mp4`)
- `--gps`: path to drone GPS log CSV (default: `drone_gps_log.csv`)
- `--terrain`: terrain file (`.jsonl` or `.csv`) (default: `terrain_data.jsonl`)
- `--threat-positions`: base threats CSV (default: `threat_positions.csv`)
- `--device`: `cuda:0` or `cpu` (auto if omitted)
- `--model`: YOLO weights (e.g., `ultralytics/yolov8l`)
- `--imgsz`: inference size (e.g., 960)
- `--conf`: detection confidence threshold (default: 0.15)
- `--iou`: NMS IoU threshold (default: 0.6)
- `--skip-frames`: process every Nth frame (default: 1)
- `--batch-size`: batch frames for inference (default: 4)
- `--max-seconds`: cap processing to first N seconds of the video
- Outputs (customize with):
  - `--raw-detections` (default: `raw_detections.csv`)
  - `--detected-persons` (default: `detected_persons.csv`)
  - `--output` (default: `Strategic_Analysis_Output.csv`)

## Outputs

- `raw_detections.csv` — per-frame YOLO boxes: `frame,x1,y1,x2,y2,x_center,y_center,confidence,class_id`
- `detected_persons.csv` — GPS coordinates for detections: `frame,latitude,longitude,confidence`
  - May also include appended coordinates from `threat_positions.csv` when > 2 entries
- `Strategic_Analysis_Output.csv` — full terrain analysis with suitability scores

## Showing Detections Live

To render boxes while detecting:
```bash
python detect_people.py \
  --video footage.mp4 \
  --imgsz 960 --conf 0.35 --iou 0.6 \
  --skip-frames 0 --max-seconds 30
```
Press `q` to quit.

## GPU Notes

- Use `--device cuda:0` if a CUDA-enabled GPU is available; FP16 is enabled automatically.
- If you see CUDA device errors, run on CPU (`--device cpu`) or fix drivers as needed.

## Troubleshooting

- GPS log length mismatch: regenerate with `create_gps_log.py` using the same `--skip-frames` as detection.
- No detections: increase `--imgsz` (e.g., 1152 or 1280) or decrease `--conf`.
- Too many false positives: increase `--conf` (0.35–0.5) and set `--skip-frames 0`.


