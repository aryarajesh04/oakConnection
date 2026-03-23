#!/usr/bin/env python3
"""
detect_laptop.py - Run door detection on a laptop webcam.

Usage:
    python detect_laptop.py
    python detect_laptop.py --camera 1 --model best.pt --fps 5

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = {0: "Closed", 1: "Open", 2: "Partially Open"}
CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (0, 165, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run door detection on a laptop webcam.")
    parser.add_argument("--model", default="best.pt", help="Path to the YOLO model file.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to open.")
    parser.add_argument("--fps", type=float, default=3.0, help="Inference FPS cap.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    return parser.parse_args()


def draw_detections(frame: np.ndarray, results, conf_threshold: float) -> np.ndarray:
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            x_center, y_center, w, h = box.xywh[0].tolist()
            x1 = max(0, int(x_center - w / 2))
            y1 = max(0, int(y_center - h / 2))
            x2 = min(frame.shape[1] - 1, int(x_center + w / 2))
            y2 = min(frame.shape[0] - 1, int(y_center + h / 2))

            cls_id = int(box.cls[0])
            label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            text_top = max(text_h + baseline + 4, y1)
            cv2.rectangle(
                frame,
                (x1, text_top - text_h - baseline - 4),
                (x1 + text_w + 4, text_top),
                color,
                -1,
            )
            cv2.putText(
                frame,
                text,
                (x1 + 2, text_top - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

    return frame


def draw_overlay(frame: np.ndarray, inference_fps: float) -> np.ndarray:
    for cls_id, label in CLASS_NAMES.items():
        color = CLASS_COLORS[cls_id]
        y_pos = 30 + cls_id * 30
        cv2.rectangle(frame, (10, y_pos - 15), (30, y_pos + 5), color, -1)
        cv2.putText(
            frame,
            label,
            (38, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        f"Inference FPS: {inference_fps:.1f}",
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
    )
    return frame


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Error: Could not open webcam at index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Running door detection on webcam {args.camera} with model {args.model}")
    print("Press 'q' to quit.")

    frame_interval = 1.0 / max(args.fps, 0.1)
    last_inference_time = 0.0
    last_results = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            now = time.time()
            if (now - last_inference_time) >= frame_interval:
                last_inference_time = now
                last_results = model(frame, verbose=False)

            display_frame = frame.copy()
            if last_results is not None:
                display_frame = draw_detections(display_frame, last_results, args.conf)
            display_frame = draw_overlay(display_frame, 1.0 / frame_interval)

            cv2.imshow("Door Detection - Laptop Webcam", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
