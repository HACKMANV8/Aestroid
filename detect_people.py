#!/usr/bin/env python3
import os
import sys
import argparse
import csv

try:
    import cv2
except Exception as e:
    print("error: opencv not installed (pip install opencv-python)")
    sys.exit(1)

try:
    from ultralytics import YOLO
except Exception as e:
    print("error: ultralytics not installed (pip install ultralytics)")
    sys.exit(1)

# torch is optional here; used for device/half decisions if present
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


PERSON_CLASS_IDS = {0, 1}  # visdrone: 0=pedestrian, 1=person


def load_model(model_arg: str | None) -> YOLO:
    # precedence: explicit path -> local visDrone.pt -> HF repo
    candidates = []
    if model_arg:
        candidates.append(model_arg)
    here = os.path.dirname(__file__)
    # common local filenames
    candidates.append(os.path.join(here, 'visDrone.pt'))
    candidates.append(os.path.join(here, 'visdrone.pt'))
    # known repos/weights (fallbacks) - USE NANO FOR SPEED
    candidates.append('Mahadih534/YoloV8-VisDrone')
    candidates.append('ultralytics/yolov8n')         # Faster nano model
    candidates.append('yolov8n.pt')                  # Ultralytics CDN

    last_err = None
    for cand in candidates:
        try:
            print(f"loading model: {cand}")
            return YOLO(cand)
        except Exception as e:
            last_err = e
            continue
    print(f"error: failed to load model. last error: {last_err}")
    print("tips:")
    print(" - place visDrone.pt next to this script or pass --model /path/to/visDrone.pt")
    print(" - or allow fallback to 'ultralytics/yolov8n' / 'yolov8n.pt' by ensuring internet access")
    sys.exit(1)


def count_people(video_path: str, model: YOLO, show: bool, max_frames: int | None,
                 imgsz: int, conf: float, iou: float, tta: bool, device: str | None,
                 upsample: int, skip_frames: int, batch_size: int,
                 out_csv: str | None = None) -> int:
    if not os.path.exists(video_path):
        print(f"error: video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: cannot open video: {video_path}")
        sys.exit(1)

    # Get video properties for progress indication
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video: {total_frames} frames @ {fps:.1f} fps")
    print(f"processing every {skip_frames + 1} frame(s) with batch size {batch_size}")

    total_count = 0
    frame_idx = 0
    processed_frames = 0

    # decide FP16 usage: enable only on CUDA when supported
    use_half = False
    if device and isinstance(device, str) and device.startswith('cuda'):
        if torch is not None:
            try:
                use_half = bool(torch.cuda.is_available())
            except Exception:
                use_half = False

    frame_batch = []
    frame_index_batch = []
    csv_writer = None
    csv_file = None
    if out_csv:
        csv_file = open(out_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame','x1','y1','x2','y2','x_center','y_center','confidence','class_id'
        ])
    
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # Stop if a time cap is set via max_frames expressed as seconds externally
        # We enforce time-based stop in the caller by converting seconds to frame limit,
        # but also defensively stop here if processed time exceeds a sentinel value

        # Skip frames for speed
        if frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            continue

        if upsample and upsample > 1:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (w * upsample, h * upsample), interpolation=cv2.INTER_CUBIC)

        frame_batch.append(frame)
        frame_index_batch.append(frame_idx)

        # Process batch when ready or at end
        if len(frame_batch) >= batch_size or (max_frames and processed_frames + len(frame_batch) >= max_frames):
            results = model.predict(
                source=frame_batch,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device if device else None,
                half=use_half,
                augment=tta,
                verbose=False,
                stream=False,  # Return list for batches
            )
            
            for i, result in enumerate(results):
                frame_count = 0
                if result.boxes is not None:
                    for b in result.boxes:
                        try:
                            cls_id = int(b.cls[0])
                            if cls_id in PERSON_CLASS_IDS:
                                frame_count += 1
                                if csv_writer is not None and i < len(frame_index_batch):
                                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu().numpy()]
                                    conf_v = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                                    cx = (x1 + x2) / 2.0
                                    cy = (y1 + y2) / 2.0
                                    csv_writer.writerow([
                                        frame_index_batch[i], x1, y1, x2, y2, cx, cy, conf_v, cls_id
                                    ])
                        except Exception:
                            continue

                total_count += frame_count
                processed_frames += 1

                if show and i < len(frame_batch):
                    annotated = result.plot()
                    cv2.putText(annotated, f'people: {frame_count} | frame: {processed_frames}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow('visdrone detection', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return total_count

            # Progress update
            if processed_frames % 30 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"progress: {progress:.1f}% ({processed_frames} frames processed, {total_count} detections)")

            frame_batch = []
            frame_index_batch = []

        frame_idx += 1
        if max_frames is not None and processed_frames >= max_frames:
            break

    # Process remaining frames in batch
    if frame_batch:
        results = model.predict(
            source=frame_batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device if device else None,
            half=use_half,
            augment=tta,
            verbose=False,
            stream=False,
        )
        
        for i, result in enumerate(results):
            frame_count = 0
            if result.boxes is not None:
                for b in result.boxes:
                    try:
                        cls_id = int(b.cls[0])
                        if cls_id in PERSON_CLASS_IDS:
                            frame_count += 1
                            if csv_writer is not None and i < len(frame_index_batch):
                                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu().numpy()]
                                conf_v = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                                cx = (x1 + x2) / 2.0
                                cy = (y1 + y2) / 2.0
                                csv_writer.writerow([
                                    frame_index_batch[i], x1, y1, x2, y2, cx, cy, conf_v, cls_id
                                ])
                    except Exception:
                        continue
            total_count += frame_count
            processed_frames += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()
    if csv_file is not None:
        csv_file.close()
    
    # Extrapolate total count based on skip rate
    extrapolated_count = total_count * (skip_frames + 1)
    print(f"processed {processed_frames} frames (skipped {skip_frames} between each)")
    print(f"raw detections: {total_count}")
    print(f"extrapolated total: {extrapolated_count}")
    
    return extrapolated_count


def main():
    parser = argparse.ArgumentParser(description='count people in a video using visdrone yolov8')
    parser.add_argument('--video', default='footage.mp4', help='path to video file')
    parser.add_argument('--model', default=None, help='model path or repo (e.g., visDrone.pt)')
    parser.add_argument('--no-gui', action='store_true', help='disable display window')
    parser.add_argument('--max-frames', type=int, default=None, help='process at most N processed frames')
    parser.add_argument('--max-seconds', type=float, default=None, help='stop after this many seconds of video (overrides --max-frames)')
    parser.add_argument('--imgsz', type=int, default=640, help='inference image size (default: 640 for speed)')
    parser.add_argument('--conf', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='iou threshold for NMS')
    parser.add_argument('--tta', action='store_true', help='enable test-time augmentation')
    parser.add_argument('--device', default=None, help='cuda:0 or cpu')
    parser.add_argument('--upsample', type=int, default=0, help='pre-scale frames by this factor before detection')
    parser.add_argument('--skip-frames', type=int, default=1, help='process every Nth frame (0=all, 1=every 2nd)')
    parser.add_argument('--batch-size', type=int, default=4, help='number of frames to process in batch')
    parser.add_argument('--out-csv', default=None, help='optional path to write raw detection boxes per frame')
    args = parser.parse_args()

    # Use small model by default (balanced speed/accuracy)
    if not args.model:
        args.model = 'ultralytics/yolov8s'
    model = load_model(args.model)

    # determine device automatically if not provided
    chosen_device = args.device
    if not chosen_device:
        if torch is not None:
            try:
                chosen_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            except Exception:
                chosen_device = 'cpu'
        else:
            chosen_device = 'cpu'

    print(f"using device: {chosen_device}")

    # move model to device and fuse for faster inference when supported
    try:
        if hasattr(model, 'to') and chosen_device:
            model.to(chosen_device)
    except Exception:
        pass
    try:
        if hasattr(model, 'fuse'):
            model.fuse()
    except Exception:
        pass
    
    # If max-seconds is provided, convert to processed frame budget using fps and skip rate
    if args.max_seconds is not None:
        # Attempt to read fps for accurate conversion
        try:
            cap_tmp = cv2.VideoCapture(args.video)
            fps_tmp = cap_tmp.get(cv2.CAP_PROP_FPS)
            cap_tmp.release()
        except Exception:
            fps_tmp = 0.0
        if fps_tmp and fps_tmp > 0:
            processed_fps = fps_tmp / float(args.skip_frames + 1)
            args.max_frames = int(processed_fps * float(args.max_seconds))

    total = count_people(
        args.video, model,
        show=not args.no_gui,
        max_frames=args.max_frames,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        tta=args.tta,
        device=chosen_device,
        upsample=args.upsample,
        skip_frames=args.skip_frames,
        batch_size=args.batch_size,
        out_csv=args.out_csv,
    )
    print(f"total_people_detections: {total}")


if __name__ == '__main__':
    main()