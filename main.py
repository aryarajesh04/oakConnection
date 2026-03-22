import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- Config ---
MODEL_PATH = "best.pt"
TARGET_FPS = 3
CONF_THRESHOLD = 0.5
CAMERA_INDEX = 0

# Class definitions
CLASS_NAMES = {0: "Closed", 1: "Open", 2: "Partially Open"}
CLASS_COLORS = {
    0: (0, 0, 255),    # Red   — Closed
    1: (0, 255, 0),    # Green — Open
    2: (0, 165, 255),  # Orange — Partially Open
}

model = YOLO(MODEL_PATH)


def draw_detections(frame: np.ndarray, results) -> np.ndarray:
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x_center, y_center, w, h = box.xywh[0].tolist()
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            cls_id = int(box.cls[0])
            label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background for readability
            text = f"{label} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5),
                          (x1 + text_w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            print(f"[{label.upper()}] Center=({x_center:.1f}, {y_center:.1f}) "
                  f"W={w:.1f} H={h:.1f} Conf={conf:.2f}")

    return frame


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Running YOLOv8 door detection on webcam at {TARGET_FPS} FPS...")
    print("Classes: 0=Closed (Red), 1=Open (Green), 2=Partially Open (Orange)")
    print("Press 'q' to quit.")

    frame_interval = 1.0 / TARGET_FPS
    last_inference_time = 0
    last_results = None

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
            display_frame = draw_detections(display_frame, last_results)

        # Legend
        for cls_id, label in CLASS_NAMES.items():
            color = CLASS_COLORS[cls_id]
            y_pos = 30 + cls_id * 30
            cv2.rectangle(display_frame, (10, y_pos - 15), (30, y_pos + 5), color, -1)
            cv2.putText(display_frame, label, (38, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # FPS indicator
        cv2.putText(display_frame, f"Inference FPS: {TARGET_FPS}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        cv2.imshow("Door Detection — Webcam", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()