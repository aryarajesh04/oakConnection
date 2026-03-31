#!/usr/bin/env python3
"""
detect_depth_nav.py — Door detection on OAK-D Lite with navigation path overlay.

Usage:
    python detect_depth_nav.py <path_to_best.blob>

Press 'q' to quit.
"""

import math
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
CONFIDENCE_THRESH = 0.50
IOU_THRESH        = 0.45
NUM_CLASSES       = 3
IMGSZ             = 640
DISPLAY_W         = 1280
DISPLAY_H         = 720
CROP_X            = (DISPLAY_W - IMGSZ) // 2   # 320
CROP_Y            = (DISPLAY_H - IMGSZ) // 2   # 40
DEPTH_MIN_MM      = 100
DEPTH_MAX_MM      = 8000
WHEELCHAIR_WIDTH_MM = 630   # physical width of wheelchair in mm

if not BLOB_PATH:
    print("Usage: python detect_depth_nav.py <path/to/best.blob>")
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
    tensor: numpy array shape (4+NUM_CLASSES, 8400) or (8400, 4+NUM_CLASSES).
    Returns list of (x1, y1, x2, y2, confidence, class_id) in pixel coords.
    """
    expected_rows = 4 + NUM_CLASSES
    if tensor.ndim == 3:
        tensor = tensor[0]

    if tensor.shape[0] != expected_rows:
        tensor = tensor.T

    scale_x = img_w / IMGSZ
    scale_y = img_h / IMGSZ

    cx = tensor[0] * scale_x
    cy = tensor[1] * scale_y
    bw = tensor[2] * scale_x
    bh = tensor[3] * scale_y

    class_scores = tensor[4:4 + NUM_CLASSES]
    class_ids    = np.argmax(class_scores, axis=0)
    confidences  = class_scores[class_ids, np.arange(class_scores.shape[1])]

    mask = confidences >= conf_thresh
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    class_ids   = class_ids[mask]
    confidences = confidences[mask]

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
# Navigation path drawing
# ---------------------------------------------------------------------------

def bezier_cubic(p0, p1, p2, p3, n=60):
    """Sample n+1 points along a cubic Bezier curve defined by four control points."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        pts.append((int(x), int(y)))
    return pts


def draw_nav_path(frame, x1, y1, x2, y2, label, color, focal_px, z_centre, angle_deg):
    """
    Overlay a navigation path from the wheelchair (bottom-centre of frame) to the door.

    - open/semi → cubic Bezier path + north arrow + alignment feedback
      - green  : door is within ±50 px of frame centre (aligned)
      - orange : door is off-centre → "TURN LEFT" or "TURN RIGHT"
    - closed   → red X + "BLOCKED"
    """
    h, w = frame.shape[:2]
    door_cx = (x1 + x2) // 2
    door_cy = (y1 + y2) // 2

    if label == "closed":
        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
        cv2.line(frame, (x2, y1), (x1, y2), color, 3)
        cv2.putText(frame, "BLOCKED", (door_cx - 42, door_cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return

    # Alignment → path colour
    aligned    = abs(door_cx - w // 2) <= 50
    path_color = (0, 220, 0) if aligned else (0, 140, 255)

    # Bezier control points
    path_height  = max(h - y2, 1)
    angle_rad    = math.radians(angle_deg) if angle_deg is not None else 0.0
    approach_dist = path_height * 0.4

    P0 = (w // 2, h)
    P1 = (w // 2, h - int(path_height * 0.45))
    P2 = (int(door_cx - math.sin(angle_rad) * approach_dist),
          int(y2      + math.cos(angle_rad) * approach_dist))
    P3 = (door_cx, int(y2))

    pts = bezier_cubic(P0, P1, P2, P3)
    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, path_color, 2)

    # North arrow — overlaid on the straight initial section, always points up
    cv2.arrowedLine(frame, (w // 2, h - 20), (w // 2, h - 70),
                    path_color, 2, tipLength=0.3)

    # Turn instruction when not aligned
    if not aligned:
        turn_text = "TURN RIGHT" if door_cx > w // 2 else "TURN LEFT"
        mid_pt    = pts[len(pts) // 2]
        (tw, _), _ = cv2.getTextSize(turn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(frame, turn_text, (mid_pt[0] - tw // 2, mid_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, path_color, 2)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
pipeline = dai.Pipeline()

# RGB camera
camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
rgbOut = camRgb.requestOutput((IMGSZ, IMGSZ), dai.ImgFrame.Type.BGR888p)

# Mono cameras for stereo depth
monoLeft  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
leftOut   = monoLeft.requestOutput((640, 480), dai.ImgFrame.Type.GRAY8)
rightOut  = monoRight.requestOutput((640, 480), dai.ImgFrame.Type.GRAY8)

# Stereo depth — aligned to RGB frame
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setSubpixel(False)
stereo.setOutputSize(IMGSZ, IMGSZ)
leftOut.link(stereo.left)
rightOut.link(stereo.right)

# Neural network (blob on VPU)
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(BLOB_PATH)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
rgbOut.link(nn.input)

qRgb   = rgbOut.createOutputQueue(maxSize=4, blocking=False)
qDet   = nn.out.createOutputQueue(maxSize=4, blocking=False)
qDepth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
pipeline.start()

# Focal length + principal point from calibration (used for true 3D width)
try:
    device     = pipeline.getDefaultDevice()
    calib      = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720)
    focal_px   = intrinsics[0][0]   # fx
    cam_cx     = intrinsics[0][2]   # principal point x
    print(f"Calibration: fx={focal_px:.1f} px  cx={cam_cx:.1f} px")
except Exception:
    HFOV_DEG = 73.0  # OAK-D Lite colour camera horizontal FOV fallback
    focal_px = (1280 / 2) / math.tan(math.radians(HFOV_DEG / 2))
    cam_cx   = 1280 / 2
    print(f"Focal length from FOV fallback ({HFOV_DEG}°): fx={focal_px:.1f} px  cx={cam_cx:.1f} px")

start          = time.monotonic()
frames         = 0
fps            = 0.0
WHITE          = (255, 255, 255)
printed_layers = False

print("Running, press 'q' to quit.")

try:
    while pipeline.isRunning():
        inRgb   = qRgb.tryGet()
        inDet   = qDet.tryGet()
        inDepth = qDepth.tryGet()

        if inRgb is None or inDet is None or inDepth is None:
            print(f"[WAIT] rgb={inRgb is not None}  det={inDet is not None}  depth={inDepth is not None}", flush=True)
            if cv2.waitKey(1) == ord("q"):
                break
            continue

        if not printed_layers:
            print("NN output layers:", inDet.getAllLayerNames())
            printed_layers = True

        frame      = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()   # uint16, values in mm
        h, w       = frame.shape[:2]

        if depthFrame.shape[:2] != (h, w):
            depthFrame = cv2.resize(depthFrame, (w, h), interpolation=cv2.INTER_NEAREST)

        # FPS
        frames += 1
        elapsed = time.monotonic() - start
        if elapsed >= 1.0:
            fps    = frames / elapsed
            frames = 0
            start  = time.monotonic()

        # Parse detections
        layer_names = inDet.getAllLayerNames()
        tensor_name = "output0" if "output0" in layer_names else (layer_names[0] if layer_names else None)
        detections  = []
        if tensor_name:
            try:
                tensor = np.array(inDet.getTensor(tensor_name), dtype=np.float32)
                if tensor.ndim == 3:
                    tensor = tensor[0]
                detections = parse_yolov8(tensor, CONFIDENCE_THRESH, IOU_THRESH, w, h)
            except Exception as e:
                print(f"[WARN] tensor parse error: {e}", flush=True)

        print(f"Detections: {len(detections)}", flush=True)

        r = 5  # depth patch half-size in pixels

        def sample_depth(px, py):
            patch = depthFrame[
                max(0, py - r):min(h, py + r),
                max(0, px - r):min(w, px + r),
            ]
            valid = patch[(patch >= DEPTH_MIN_MM) & (patch <= DEPTH_MAX_MM)]
            return int(np.median(valid)) if valid.size else 0

        # ------------------------------------------------------------------
        # Pass 1 — collect per-detection data; keep only highest-confidence   
        # ------------------------------------------------------------------
        det_data = []

        # Filter to only the single highest-confidence detection                                                
        if detections:                                                                                          
            detections = [max(detections, key=lambda d: d[4])]  

        for (bx1, by1, bx2, by2, conf, label_idx) in detections:
            x1 = max(0, min(int(bx1), w - 1))
            y1 = max(0, min(int(by1), h - 1))
            x2 = max(0, min(int(bx2), w - 1))
            y2 = max(0, min(int(by2), h - 1))

            label = LABEL_MAP[label_idx] if label_idx < len(LABEL_MAP) else str(label_idx)
            color = LABEL_COLORS.get(label, WHITE)

            cy_px = (y1 + y2) // 2
            cx_px = (x1 + x2) // 2

            z_centre = sample_depth(cx_px, cy_px)
            z_left   = sample_depth(x1,    cy_px)
            z_right  = sample_depth(x2,    cy_px)

            # True 3D width
            if z_left and z_right:
                X_left   = (x1 - cam_cx) * z_left  / focal_px
                X_right  = (x2 - cam_cx) * z_right / focal_px
                width_mm = int(math.sqrt((X_right - X_left) ** 2 + (z_right - z_left) ** 2))
            elif z_centre:
                width_mm = int((x2 - x1) * z_centre / focal_px)
            else:
                width_mm = 0

            # Angle of door relative to camera
            if width_mm and z_left and z_right:
                sin_val   = max(-1.0, min(1.0, (z_right - z_left) / width_mm))
                angle_deg = math.degrees(math.asin(sin_val))
            else:
                angle_deg = None

            det_data.append(dict(
                x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                label=label, color=color,
                z_centre=z_centre, z_left=z_left, z_right=z_right,
                width_mm=width_mm, angle_deg=angle_deg,
            ))

            angle_str = f"{angle_deg:+.1f}°" if angle_deg is not None else "--"
            print(f"  {label} {int(conf*100)}%  x1={bx1:.1f} y1={by1:.1f} x2={bx2:.1f} y2={by2:.1f}  "
                  f"Z_left={z_left}mm  Z_centre={z_centre}mm  Z_right={z_right}mm  "
                  f"W={width_mm}mm ({width_mm/1000:.2f}m)  angle={angle_str}", flush=True)

            L = 20  # corner bracket length in pixels
            cv2.line(frame, (x1, y1), (x1 + L, y1), color, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + L), color, 2)
            cv2.line(frame, (x2, y1), (x2 - L, y1), color, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + L), color, 2)
            cv2.line(frame, (x1, y2), (x1 + L, y2), color, 2)
            cv2.line(frame, (x1, y2), (x1, y2 - L), color, 2)
            cv2.line(frame, (x2, y2), (x2 - L, y2), color, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - L), color, 2)
            for i, text in enumerate([
                f"{label}  {int(conf * 100)}%",
                f"Z: L={z_left}  C={z_centre}  R={z_right}mm" if z_centre else "Z: --",
                f"W: {width_mm}mm  angle: {angle_str}" if width_mm else "W: --",
            ]):
                cv2.putText(frame, text, (x1 + 6, y1 + 18 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        # ------------------------------------------------------------------
        # Pass 2 — draw nav path for nearest open/semi door only;
        #          draw X/BLOCKED for every closed door
        # ------------------------------------------------------------------
        best       = None
        best_score = float("inf")
        for d in det_data:
            if d["label"] not in ("open", "semi"):
                continue
            score = d["z_centre"] if d["z_centre"] > 0 else \
                    1.0 / max((d["x2"] - d["x1"]) * (d["y2"] - d["y1"]), 1)
            if score < best_score:
                best_score = score
                best = d

        if best is not None:
            draw_nav_path(frame,
                          best["x1"], best["y1"], best["x2"], best["y2"],
                          best["label"], best["color"],
                          focal_px, best["z_centre"], best["angle_deg"])

        for d in det_data:
            if d["label"] == "closed":
                draw_nav_path(frame,
                              d["x1"], d["y1"], d["x2"], d["y2"],
                              d["label"], d["color"],
                              focal_px, d["z_centre"], d["angle_deg"])

        cv2.putText(frame, f"FPS: {fps:.1f}", (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Door Detection + Nav", display)

        if cv2.waitKey(1) == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
