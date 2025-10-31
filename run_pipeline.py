#!/usr/bin/env python3
import os
import sys
import csv
import math
import argparse
import subprocess
import pandas as pd

try:
    import cv2
except Exception:
    print("error: opencv not installed (pip install opencv-python)")
    sys.exit(1)


def run_detection(video_path: str, out_raw_csv: str, device: str | None,
                  model: str | None, imgsz: int, conf: float, iou: float,
                  skip_frames: int, batch_size: int, max_seconds: float | None) -> None:
    script = os.path.join(os.path.dirname(__file__), 'detect_people.py')
    cmd = [sys.executable, script,
           '--video', video_path,
           '--no-gui',
           '--imgsz', str(imgsz),
           '--conf', str(conf),
           '--iou', str(iou),
           '--skip-frames', str(skip_frames),
           '--batch-size', str(batch_size),
           '--out-csv', out_raw_csv]
    if device:
        cmd += ['--device', device]
    if model:
        cmd += ['--model', model]
    if max_seconds is not None:
        cmd += ['--max-seconds', str(max_seconds)]
    print('running detection:', ' '.join(cmd))
    subprocess.check_call(cmd)


def pixel_to_gps(center_x: float, center_y: float, frame_w: int, frame_h: int,
                 drone_lat: float, drone_lon: float, drone_alt_m: float,
                 camera_fov_deg: float) -> tuple[float, float]:
    ground_width_m = 2.0 * drone_alt_m * math.tan(math.radians(camera_fov_deg / 2.0))
    meters_per_pixel_x = ground_width_m / float(frame_w)
    meters_per_pixel_y = ground_width_m / float(frame_h)
    offset_x_m = (center_x - (frame_w / 2.0)) * meters_per_pixel_x
    offset_y_m = ((frame_h / 2.0) - center_y) * meters_per_pixel_y
    lat_offset_deg = offset_y_m / 111111.0
    lon_offset_deg = offset_x_m / (111111.0 * math.cos(math.radians(drone_lat)))
    return drone_lat + lat_offset_deg, drone_lon + lon_offset_deg


def convert_raw_to_gps(raw_csv: str, gps_csv: str, video_path: str,
                       out_people_csv: str, min_conf: float,
                       camera_fov_deg: float, default_alt_m: float) -> None:
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(raw_csv)
    gps_df = pd.read_csv(gps_csv)
    if 'altitude' not in gps_df.columns:
        gps_df['altitude'] = default_alt_m
    gps_df = gps_df[['frame','latitude','longitude','altitude']]
    gps_map = {int(r.frame): (float(r.latitude), float(r.longitude), float(r.altitude))
               for r in gps_df.itertuples(index=False)}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open video: {video_path}')
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    rows_out = []
    with open(raw_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row['confidence']) if 'confidence' in row else 0.0
            if conf < min_conf:
                continue
            frame = int(row['frame'])
            if frame not in gps_map:
                continue
            lat, lon, alt = gps_map[frame]
            cx = float(row['x_center'])
            cy = float(row['y_center'])
            person_lat, person_lon = pixel_to_gps(cx, cy, frame_w, frame_h, lat, lon, alt, camera_fov_deg)
            rows_out.append({
                'frame': frame,
                'latitude': person_lat,
                'longitude': person_lon,
                'confidence': conf,
            })

    if not rows_out:
        print('warning: no person detections passed the filters; writing empty CSV')
    pd.DataFrame(rows_out).to_csv(out_people_csv, index=False, float_format='%.8f')
    print(f'wrote {len(rows_out)} detections -> {out_people_csv}')


def run_ml(terrain: str, threats: str, detected_people: str, output: str,
           min_person_confidence: float) -> None:
    script = os.path.join(os.path.dirname(__file__), 'ml.py')
    cmd = [sys.executable, script,
           '--terrain', terrain,
           '--threat-positions', threats,
           '--detected-persons', detected_people,
           '--min-person-confidence', str(min_person_confidence),
           '--output', output]
    print('running ml:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser(description='Run detection -> GPS conversion -> ML analysis pipeline')
    p.add_argument('--video', default='footage.mp4')
    p.add_argument('--gps', default='drone_gps_log.csv')
    p.add_argument('--terrain', default='terrain_data.jsonl')
    p.add_argument('--threat-positions', default='threat_positions.csv')
    p.add_argument('--device', default=None)
    p.add_argument('--model', default=None)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.15)
    p.add_argument('--iou', type=float, default=0.6)
    p.add_argument('--skip-frames', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-seconds', type=float, default=None, help='only process the first N seconds of video')
    p.add_argument('--min-person-confidence', type=float, default=0.5)
    p.add_argument('--camera-fov-deg', type=float, default=84.0)
    p.add_argument('--default-alt-m', type=float, default=100.0)
    p.add_argument('--raw-detections', default='raw_detections.csv')
    p.add_argument('--detected-persons', default='detected_persons.csv')
    p.add_argument('--output', default='Strategic_Analysis_Output.csv')
    args = p.parse_args()

    run_detection(
        video_path=args.video,
        out_raw_csv=args.raw_detections,
        device=args.device,
        model=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        skip_frames=args.skip_frames,
        batch_size=args.batch_size,
        max_seconds=args.max_seconds,
    )

    convert_raw_to_gps(
        raw_csv=args.raw_detections,
        gps_csv=args.gps,
        video_path=args.video,
        out_people_csv=args.detected_persons,
        min_conf=args.min_person_confidence,
        camera_fov_deg=args.camera_fov_deg,
        default_alt_m=args.default_alt_m,
    )

    run_ml(
        terrain=args.terrain,
        threats=args.threat_positions,
        detected_people=args.detected_persons,
        output=args.output,
        min_person_confidence=args.min_person_confidence,
    )


if __name__ == '__main__':
    main()


