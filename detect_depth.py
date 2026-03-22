#!/usr/bin/env python3
"""
detect_depth.py — Door detection with spatial depth on Luxonis OAK camera.

Usage:
    python detect_depth.py <path_to_best.blob>

Press 'q' to quit.
"""

import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOB_PATH         = sys.argv[1] if len(sys.argv) > 1 else None
LABEL_MAP         = ["closed", "open", "semi"]
LABEL_COLORS      = {
    "closed": (0,   0,   220),
    "open":   (0,   220, 0  ),
    "semi":   (0,   200, 255),
}
CONFIDENCE_THRESH = 0.05
IOU_THRESH        = 0.45
NUM_CLASSES       = 3
IMGSZ             = 640
DEPTH_MIN_MM      = 100
DEPTH_MAX_MM      = 8000

if not BLOB_PATH:
    print("Usage: python detect_depth.py <path/to/best.blob>")
    sys.exit(1)

if not Path(BLOB_PATH).exists():
    print(f"[ERROR] Blob not found: {BLOB_PATH}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# YOLO v8 host-side parsing helpers
# ---------------------------------------------------------------------------

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes, dtype=float)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_   = np.maximum(0, xx2 - xx1)
        h_   = np.maximum(0, yy2 - yy1)
        iou  = (w_ * h_) / (areas[i] + areas[order[1:]] - w_ * h_ + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def parse_yolov8(tensor, conf_thresh, iou_thresh, img_w, img_h):
    """
    tensor: numpy array of shape (4+NUM_CLASSES, 8400) or (8400, 4+NUM_CLASSES).
    Returns list of (x1, y1, x2, y2, confidence, class_id) in pixel coords.
    """
    expected_rows = 4 + NUM_CLASSES
    if tensor.ndim == 1:
        try:
            tensor = tensor.reshape(expected_rows, -1)
        except ValueError:
            tensor = tensor.reshape(-1, expected_rows).T

    if tensor.shape[0] != expected_rows:
        tensor = tensor.T  # flip if needed

    scale_x = img_w / IMGSZ
    scale_y = img_h / IMGSZ

    cx = tensor[0] * scale_x
    cy = tensor[1] * scale_y
    bw = tensor[2] * scale_x
    bh = tensor[3] * scale_y

    class_scores = tensor[4:4 + NUM_CLASSES]           # (NUM_CLASSES, 8400)
    class_ids    = np.argmax(class_scores, axis=0)
    confidences  = class_scores[class_ids, np.arange(class_scores.shape[1])]

    mask = confidences >= conf_thresh
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    class_ids      = class_ids[mask]
    confidences    = confidences[mask]

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    results = []
    for cls in range(NUM_CLASSES):
        cls_mask = class_ids == cls
        if not np.any(cls_mask):
            continue
        ix   = np.where(cls_mask)[0]
        keep = nms(
            np.stack([x1[ix], y1[ix], x2[ix], y2[ix]], axis=1),
            confidences[ix],
            iou_thresh,
        )
        for k in keep:
            results.append((
                float(x1[ix[k]]), float(y1[ix[k]]),
                float(x2[ix[k]]), float(y2[ix[k]]),
                float(confidences[ix[k]]),
                int(class_ids[ix[k]]),
            ))
    return results


# ---------------------------------------------------------------------------
# Pipeline (depthai 3.x)
# ---------------------------------------------------------------------------
pipeline = dai.Pipeline()

# Cameras
camRgb    = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
monoLeft  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

rgbOut   = camRgb.requestOutput((IMGSZ, IMGSZ), dai.ImgFrame.Type.BGR888p)
leftOut  = monoLeft.requestOutput((640, 400),   dai.ImgFrame.Type.GRAY8)
rightOut = monoRight.requestOutput((640, 400),  dai.ImgFrame.Type.GRAY8)

# Stereo depth
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setSubpixel(False)
stereo.setOutputSize(IMGSZ, IMGSZ)
leftOut.link(stereo.left)
rightOut.link(stereo.right)

# Neural network (plain — no built-in parser)
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(BLOB_PATH)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
rgbOut.link(nn.input)

# Output queues
qRgb   = rgbOut.createOutputQueue(maxSize=4, blocking=False)
qDet   = nn.out.createOutputQueue(maxSize=4, blocking=False)
qDepth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
pipeline.start()

start          = time.monotonic()
frames         = 0
fps            = 0.0
WHITE          = (255, 255, 255)
printed_layers = False

print("Running — press 'q' to quit.")

try:
    while pipeline.isRunning():
        inRgb   = qRgb.tryGet()
        inDet   = qDet.tryGet()
        inDepth = qDepth.tryGet()

        if inRgb is None or inDet is None or inDepth is None:
            if cv2.waitKey(1) == ord("q"):
                break
            continue

        # Print layer info once
        if not printed_layers:
            print("NN output layers:", inDet.getAllLayerNames())
            print("NNData methods:", [m for m in dir(inDet) if not m.startswith('_')])
            printed_layers = True

        frame      = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()
        h, w       = frame.shape[:2]

        # Resize depth to match RGB if needed
        if depthFrame.shape[0] != h or depthFrame.shape[1] != w:
            depthFrame = cv2.resize(depthFrame, (w, h), interpolation=cv2.INTER_NEAREST)

        # FPS
        frames += 1
        elapsed = time.monotonic() - start
        if elapsed > 1.0:
            fps    = frames / elapsed
            frames = 0
            start  = time.monotonic()

        # Parse YOLO output tensor
        layer_names = inDet.getAllLayerNames()
        tensor_name = "output0" if "output0" in layer_names else (layer_names[0] if layer_names else None)
        detections  = []
        if tensor_name:
            try:
                tensor = np.array(inDet.getTensor(tensor_name), dtype=np.float32)
                # Remove batch dimension if present: (1, 7, 8400) → (7, 8400)
                if tensor.ndim == 3:
                    tensor = tensor[0]
                # Debug on first 3 frames
                if frames <= 3:
                    t2 = tensor if tensor.shape[0] == 4 + NUM_CLASSES else tensor.T
                    scores = t2[4:4 + NUM_CLASSES]
                    print(f"[DBG] shape={tensor.shape} max_conf={float(scores.max()):.4f}", flush=True)
                detections = parse_yolov8(tensor, CONFIDENCE_THRESH, IOU_THRESH, w, h)
            except Exception as e:
                print(f"[WARN] tensor parse error: {e}", flush=True)

        print(f"Detections: {len(detections)}", flush=True)

        # Colour-map depth for display
        d_down  = depthFrame[::4]
        nonzero = d_down[d_down != 0]
        d_min   = float(np.percentile(nonzero, 1)) if nonzero.size else 0
        d_max   = float(np.percentile(d_down, 99)) if d_down.size else 1
        depthVis = np.interp(depthFrame, (d_min, d_max), (0, 255)).astype(np.uint8)
        depthVis = cv2.applyColorMap(depthVis, cv2.COLORMAP_HOT)

        for (bx1, by1, bx2, by2, conf, label_idx) in detections:
            x1 = max(0, min(int(bx1), w - 1))
            y1 = max(0, min(int(by1), h - 1))
            x2 = max(0, min(int(bx2), w - 1))
            y2 = max(0, min(int(by2), h - 1))

            label = LABEL_MAP[label_idx] if label_idx < len(LABEL_MAP) else str(label_idx)
            color = LABEL_COLORS.get(label, WHITE)

            # Sample depth at bbox centre (5 px radius)
            cx_px = (x1 + x2) // 2
            cy_px = (y1 + y2) // 2
            r     = 5
            patch = depthFrame[
                max(0, cy_px - r):min(h, cy_px + r),
                max(0, cx_px - r):min(w, cx_px + r),
            ]
            valid = patch[(patch >= DEPTH_MIN_MM) & (patch <= DEPTH_MAX_MM)]
            z_mm  = int(np.median(valid)) if valid.size else 0

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(depthVis, (x1, y1), (x2, y2), WHITE, 1)
            for i, text in enumerate([
                f"{label}  {int(conf * 100)}%",
                f"Z: {z_mm} mm  ({z_mm/1000:.2f} m)",
            ]):
                cv2.putText(frame, text, (x1 + 6, y1 + 18 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        cv2.putText(frame, f"FPS: {fps:.1f}", (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        cv2.imshow("Door Detection", frame)
        cv2.imshow("Depth", depthVis)

        if cv2.waitKey(1) == ord("q"):
            break
finally:
    pipeline.stop()
