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
DEPTH_MIN_MM      = 100
DEPTH_MAX_MM      = 8000
WHEELCHAIR_WIDTH_MM = 630   # physical width of wheelchair in mm
SCREEN_W          = 720
SCREEN_H          = 1280
CAM_W             = 640
CAM_H             = 640
CAM_X             = 40
CAM_Y             = 0
WIDTH_BUFFER_MM   = 50
PASS_WIDTH_MM     = WHEELCHAIR_WIDTH_MM + WIDTH_BUFFER_MM
ALIGN_TOLERANCE_PX = 50
WINDOW_NAME       = "Door Detection + Nav"
CLOSE_BTN_W       = 88
CLOSE_BTN_H       = 88
CLOSE_BTN_PAD     = 18

MOBILENET_BLOB  = sys.argv[2] if len(sys.argv) > 2 else str(
    (Path(__file__).parent / "../models/mobilenet-ssd_openvino_2021.4_6shave.blob").resolve()
)
PERSON_TRACKING = Path(MOBILENET_BLOB).exists()
PERSON_COLOR    = (0, 200, 255)   # BGR yellow-amber for person boxes
if not PERSON_TRACKING:
    print(f"[INFO] MobileNet blob not found -- person tracking disabled. ({MOBILENET_BLOB})")

WHITE             = (255, 255, 255)
UI_BG             = (18, 24, 34)
UI_PANEL          = (28, 36, 48)
UI_PANEL_DARK     = (20, 26, 36)
UI_BORDER         = (68, 80, 96)
UI_MUTED          = (158, 170, 184)
UI_NEUTRAL        = (110, 120, 132)
TURN_COLOR        = (0, 140, 255)
INFO_COLOR        = (255, 220, 0)

close_button_rect = (0, 0, 0, 0)
exit_requested    = False

EMA_ALPHA  = 0.05   # smoothing factor (0=frozen, 1=no smoothing); ~20-frame window at 15 fps
_ema_width = None   # smoothed width_mm for display
_ema_depth = None   # smoothed z_centre for display

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


def draw_nav_path(frame, x1, y1, x2, y2, label, color, focal_px, z_centre, _angle_deg):
    """
    Overlay a reverse-camera-style navigation path.

    - Straight path (blue filled trapezoid + red borders): current heading direction.
    - Ideal path (orange bezier curves): the path to steer toward the door.
    - closed → red X + "BLOCKED"
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

    aligned = abs(door_cx - w // 2) <= ALIGN_TOLERANCE_PX
    cx = w // 2

    PATH_ORANGE = (0, 140, 255)  # BGR — orange ideal-path guides
    arrow_color = LABEL_COLORS["open"] if aligned else PATH_ORANGE

    # ── 1. Small upward arrow showing current heading ─────────────────────────
    cv2.arrowedLine(frame, (cx, h - 20), (cx, h - 70), arrow_color, 2, tipLength=0.3)

    # ── 2. Ideal curved path: orange bezier guides toward the door ────────────
    # Show the user the corridor they need to steer into to reach the door.
    door_half_w = max(40, (x2 - x1) // 2)
    forward_dist = h - y2

    # Left orange guide: bottom-left → left edge of door
    PL0 = (cx - door_half_w, h)
    PL1 = (cx - door_half_w + int((x1 - (cx - door_half_w)) * 0.9), h - int(forward_dist * 0.25))
    PL2 = (x1, y2 + int(forward_dist * 0.6))
    PL3 = (x1, y2)
    pts_l = bezier_cubic(PL0, PL1, PL2, PL3)
    cv2.polylines(frame, [np.array(pts_l, dtype=np.int32)], False, PATH_ORANGE, 3)

    # Right orange guide: bottom-right → right edge of door
    PR0 = (cx + door_half_w, h)
    PR1 = (cx + door_half_w + int((x2 - (cx + door_half_w)) * 0.9), h - int(forward_dist * 0.25))
    PR2 = (x2, y2 + int(forward_dist * 0.6))
    PR3 = (x2, y2)
    pts_r = bezier_cubic(PR0, PR1, PR2, PR3)
    cv2.polylines(frame, [np.array(pts_r, dtype=np.int32)], False, PATH_ORANGE, 3)

    # Turn instruction when not aligned
    if not aligned:
        turn_text = "TURN RIGHT" if door_cx > cx else "TURN LEFT"
        (tw_px, _), _ = cv2.getTextSize(turn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        mid_pt = pts_l[len(pts_l) // 2] if door_cx < cx else pts_r[len(pts_r) // 2]
        cv2.putText(frame, turn_text, (cx - tw_px // 2, mid_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, PATH_ORANGE, 2)


def detection_score(det):
    """Return a lower-is-better score for choosing the most relevant doorway."""
    if det["z_centre"] > 0:
        return float(det["z_centre"])
    area = max((det["x2"] - det["x1"]) * (det["y2"] - det["y1"]), 1)
    return 1.0 / area


def select_primary_target(det_data):
    """Prefer the nearest open/semi door, then fall back to the nearest closed door."""
    best_open = None
    best_open_score = float("inf")
    best_closed = None
    best_closed_score = float("inf")

    for det in det_data:
        score = detection_score(det)
        if det["label"] in ("open", "semi") and score < best_open_score:
            best_open = det
            best_open_score = score
        elif det["label"] == "closed" and score < best_closed_score:
            best_closed = det
            best_closed_score = score

    return best_open if best_open is not None else best_closed


def fit_text_scale(text, max_width, base_scale, thickness):
    """Shrink text until it fits inside max_width."""
    scale = base_scale
    while scale > 0.45:
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if text_w <= max_width:
            return scale
        scale -= 0.05
    return 0.45


def draw_status_card(canvas, x, y, w, h, card):
    """Render a dashboard card with a colored accent rail."""
    color = card["color"]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), UI_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + 14, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), UI_BORDER, 2)

    cv2.putText(canvas, card["title"].upper(), (x + 30, y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI_MUTED, 2)

    value_scale = fit_text_scale(card["value"], w - 56, 1.45, 3)
    value_y = y + 116
    cv2.putText(canvas, card["value"], (x + 30, value_y),
                cv2.FONT_HERSHEY_SIMPLEX, value_scale, WHITE, 3)

    subtitle_scale = fit_text_scale(card["subtitle"], w - 56, 0.72, 2)
    cv2.putText(canvas, card["subtitle"], (x + 30, y + h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, UI_MUTED, 2)


def ema_update(prev, new_val, alpha=EMA_ALPHA):
    """EMA step. Holds last good value when new_val is 0 (bad depth read)."""
    if new_val <= 0:
        return prev
    if prev is None:
        return float(new_val)
    return alpha * new_val + (1.0 - alpha) * prev


def drain_latest(queue):
    """Read every pending packet and return only the newest one."""
    latest = None
    while True:
        packet = queue.tryGet()
        if packet is None:
            return latest
        latest = packet


def draw_close_button(canvas):
    """Draw a large touchscreen-friendly close button in the top-right corner."""
    global close_button_rect

    x2 = SCREEN_W - CLOSE_BTN_PAD
    y1 = CLOSE_BTN_PAD
    x1 = x2 - CLOSE_BTN_W
    y2 = y1 + CLOSE_BTN_H
    close_button_rect = (x1, y1, x2, y2)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), UI_PANEL, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), LABEL_COLORS["closed"], 3)
    cv2.line(canvas, (x1 + 24, y1 + 24), (x2 - 24, y2 - 24), WHITE, 4)
    cv2.line(canvas, (x2 - 24, y1 + 24), (x1 + 24, y2 - 24), WHITE, 4)


def handle_mouse(event, x, y, flags, param):
    """Allow the fullscreen UI to close from a touch or mouse click."""
    del flags, param
    global exit_requested

    if event != cv2.EVENT_LBUTTONUP:
        return

    x1, y1, x2, y2 = close_button_rect
    if x1 <= x <= x2 and y1 <= y <= y2:
        exit_requested = True


def build_status_summary(target, frame_width):
    """Translate one doorway target into dashboard cards and overlay pills."""
    if target is None:
        cards = {
            "door": {
                "title": "Door",
                "value": "NO DOOR",
                "subtitle": "No doorway detected yet",
                "color": UI_NEUTRAL,
            },
            "clearance": {
                "title": "Clearance",
                "value": "UNKNOWN",
                "subtitle": f"Need at least {PASS_WIDTH_MM} mm",
                "color": UI_NEUTRAL,
            },
            "guidance": {
                "title": "Guidance",
                "value": "STOP",
                "subtitle": "No target door in view",
                "color": LABEL_COLORS["closed"],
            },
            "distance": {
                "title": "Distance",
                "value": "NO DEPTH",
                "subtitle": "Waiting for a valid depth sample",
                "color": UI_NEUTRAL,
            },
        }
        return {
            "target": None,
            "cards": cards,
            "message": "No passable door detected",
            "message_color": TURN_COLOR,
        }

    label = target["label"]
    width_mm = target["width_mm"]
    z_centre = target["z_centre"]
    door_center_x = (target["x1"] + target["x2"]) // 2
    offset_px = door_center_x - frame_width // 2

    if label == "semi":
        door_value = "SEMI"
    else:
        door_value = label.upper()

    door_card = {
        "title": "Door",
        "value": door_value,
        "subtitle": f"{int(target['conf'] * 100)}% confidence",
        "color": LABEL_COLORS.get(label, UI_NEUTRAL),
    }

    if label == "closed":
        clearance_card = {
            "title": "Clearance",
            "value": "BLOCKED",
            "subtitle": "Door is closed",
            "color": LABEL_COLORS["closed"],
        }
    elif width_mm <= 0:
        clearance_card = {
            "title": "Clearance",
            "value": "UNKNOWN",
            "subtitle": f"Need at least {PASS_WIDTH_MM} mm",
            "color": UI_NEUTRAL,
        }
    elif width_mm >= PASS_WIDTH_MM:
        clearance_card = {
            "title": "Clearance",
            "value": "PASS",
            "subtitle": f"{width_mm} mm clear opening",
            "color": LABEL_COLORS["open"],
        }
    else:
        clearance_card = {
            "title": "Clearance",
            "value": "TOO NARROW",
            "subtitle": f"{width_mm} mm clear, need {PASS_WIDTH_MM} mm",
            "color": TURN_COLOR,
        }

    if label == "closed":
        guidance_card = {
            "title": "Guidance",
            "value": "STOP",
            "subtitle": "Blocked doorway",
            "color": LABEL_COLORS["closed"],
        }
    elif abs(offset_px) <= ALIGN_TOLERANCE_PX:
        guidance_card = {
            "title": "Guidance",
            "value": "CENTERED",
            "subtitle": f"Aligned within {ALIGN_TOLERANCE_PX} px",
            "color": LABEL_COLORS["open"],
        }
    else:
        guidance_card = {
            "title": "Guidance",
            "value": "TURN RIGHT" if offset_px > 0 else "TURN LEFT",
            "subtitle": f"{abs(offset_px)} px off centre",
            "color": TURN_COLOR,
        }

    display_z = target.get("display_z_centre", z_centre)
    if display_z > 0:
        distance_card = {
            "title": "Distance",
            "value": f"{display_z / 1000:.2f} m",
            "subtitle": "Door centre depth",
            "color": INFO_COLOR,
        }
    else:
        distance_card = {
            "title": "Distance",
            "value": "NO DEPTH",
            "subtitle": "Waiting for a valid depth sample",
            "color": UI_NEUTRAL,
        }

    if label == "closed":
        message = "No passable door detected"
        message_color = LABEL_COLORS["closed"]
    elif clearance_card["value"] == "PASS":
        message = "Passable door detected"
        message_color = LABEL_COLORS["open"]
    elif clearance_card["value"] == "TOO NARROW":
        message = "Opening is too narrow for the wheelchair"
        message_color = TURN_COLOR
    else:
        message = "Door found, but width is unknown"
        message_color = UI_MUTED

    cards = {
        "door": door_card,
        "clearance": clearance_card,
        "guidance": guidance_card,
        "distance": distance_card,
    }

    return {
        "target": target,
        "cards": cards,
        "message": message,
        "message_color": message_color,
    }


def render_portrait_display(camera_frame, summary, fps):
    """Place the 640x640 camera view into a 720x1280 portrait dashboard layout."""
    if camera_frame.shape[1] != CAM_W or camera_frame.shape[0] != CAM_H:
        camera_frame = cv2.resize(camera_frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((SCREEN_H, SCREEN_W, 3), UI_BG, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (SCREEN_W, CAM_H + 36), UI_PANEL_DARK, -1)
    canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = camera_frame
    cv2.rectangle(canvas, (CAM_X - 3, CAM_Y), (CAM_X + CAM_W + 3, CAM_Y + CAM_H + 6), UI_BORDER, 2)

    dash_top = CAM_Y + CAM_H + 38
    dash_left = 28
    gap = 20
    card_w = (SCREEN_W - 2 * dash_left - gap) // 2
    card_h = 220
    row1_y = dash_top
    row2_y = row1_y + card_h + gap

    cv2.putText(canvas, "DOORWAY STATUS", (dash_left, dash_top - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, UI_MUTED, 2)

    draw_status_card(canvas, dash_left, row1_y, card_w, card_h, summary["cards"]["door"])
    draw_status_card(canvas, dash_left + card_w + gap, row1_y, card_w, card_h, summary["cards"]["clearance"])
    draw_status_card(canvas, dash_left, row2_y, card_w, card_h, summary["cards"]["guidance"])
    draw_status_card(canvas, dash_left + card_w + gap, row2_y, card_w, card_h, summary["cards"]["distance"])

    footer_y = row2_y + card_h + 56
    cv2.line(canvas, (dash_left, footer_y - 26), (SCREEN_W - dash_left, footer_y - 26), UI_BORDER, 1)
    message_scale = fit_text_scale(summary["message"], SCREEN_W - 220, 0.8, 2)
    cv2.putText(canvas, summary["message"], (dash_left, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, message_scale, summary["message_color"], 2)
    cv2.putText(canvas, f"FPS {fps:.1f}", (SCREEN_W - 150, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(canvas, f"Chair {WHEELCHAIR_WIDTH_MM} mm + {WIDTH_BUFFER_MM} mm buffer",
                (dash_left, footer_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_MUTED, 2)

    draw_close_button(canvas)

    return canvas


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

qRgb   = rgbOut.createOutputQueue(maxSize=1, blocking=False)
qDet   = nn.out.createOutputQueue(maxSize=1, blocking=False)
qDepth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

# ── Person-tracking branch (parallel to door-detection branch) ────────────
if PERSON_TRACKING:
    personManip = pipeline.create(dai.node.ImageManip)
    personManip.initialConfig.setOutputSize(300, 300)
    personManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    personManip.setMaxOutputFrameSize(300 * 300 * 3)
    personManip.inputImage.setBlocking(False)
    personManip.inputImage.setMaxSize(1)

    personNN = pipeline.create(dai.node.MobileNetDetectionNetwork)
    personNN.setBlobPath(MOBILENET_BLOB)
    personNN.setConfidenceThreshold(0.5)
    personNN.setNumInferenceThreads(1)
    personNN.input.setBlocking(False)
    personNN.input.setMaxSize(1)

    personTracker = pipeline.create(dai.node.ObjectTracker)
    personTracker.setDetectionLabelsToTrack([15])  # 15 = person in MobileNet-SSD
    personTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    personTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
    try:
        personTracker.setOcclusionRatioThreshold(0.4)
    except AttributeError:
        pass
    try:
        personTracker.setTrackletMaxLifespan(120)
    except AttributeError:
        pass
    try:
        personTracker.setTrackletBirthThreshold(3)
    except AttributeError:
        pass

    # rgbOut fans out to personManip (already linked to nn.input and qRgb)
    rgbOut.link(personManip.inputImage)
    personManip.out.link(personNN.input)
    personNN.passthrough.link(personTracker.inputTrackerFrame)
    personNN.passthrough.link(personTracker.inputDetectionFrame)
    personNN.out.link(personTracker.inputDetections)

    qTracklets = personTracker.out.createOutputQueue(maxSize=1, blocking=False)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
pipeline.start()

# Focal length + principal point from calibration (used for true 3D width)
try:
    device     = pipeline.getDefaultDevice()
    calib      = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, IMGSZ, IMGSZ)
    focal_px   = intrinsics[0][0]   # fx
    cam_cx     = intrinsics[0][2]   # principal point x
    print(f"Calibration: fx={focal_px:.1f} px  cx={cam_cx:.1f} px")
except Exception:
    HFOV_DEG = 81.0  # horizontal FOV fallback when calibration is unavailable
    focal_px = (IMGSZ / 2) / math.tan(math.radians(HFOV_DEG / 2))
    cam_cx   = IMGSZ / 2
    print(f"Focal length from FOV fallback ({HFOV_DEG}°): fx={focal_px:.1f} px  cx={cam_cx:.1f} px")

start          = time.monotonic()
frames         = 0
fps            = 0.0
printed_layers = False

print("Running, press 'q' to quit.")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
cv2.setMouseCallback(WINDOW_NAME, handle_mouse)
try:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
except (cv2.error, AttributeError):
    pass

last_rgb = None
last_det = None
last_depth = None
last_display_frame = None
last_tracklets = None

try:
    while pipeline.isRunning():
        newest_rgb = drain_latest(qRgb)
        newest_det = drain_latest(qDet)
        newest_depth = drain_latest(qDepth)
        if newest_rgb is not None:
            last_rgb = newest_rgb
        if newest_det is not None:
            last_det = newest_det
        if newest_depth is not None:
            last_depth = newest_depth
        if PERSON_TRACKING:
            newest_tracklets = drain_latest(qTracklets)
            if newest_tracklets is not None:
                last_tracklets = newest_tracklets

        if exit_requested:
            break

        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except (cv2.error, AttributeError):
            pass

        if last_rgb is None or last_det is None or last_depth is None:
            wait_frame = np.full((CAM_H, CAM_W, 3), UI_PANEL_DARK, dtype=np.uint8)
            summary = build_status_summary(None, CAM_W)
            summary["message"] = "Starting camera and doorway detection"
            summary["message_color"] = UI_MUTED
            display_frame = render_portrait_display(wait_frame, summary, fps)
            last_display_frame = display_frame
            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) == ord("q"):
                break
            continue

        if not printed_layers:
            print("NN output layers:", last_det.getAllLayerNames())
            printed_layers = True

        frame      = last_rgb.getCvFrame()
        depthFrame = last_depth.getFrame()   # uint16, values in mm
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
        layer_names = last_det.getAllLayerNames()
        tensor_name = "output0" if "output0" in layer_names else (layer_names[0] if layer_names else None)
        detections  = []
        if tensor_name:
            try:
                tensor = np.array(last_det.getTensor(tensor_name), dtype=np.float32)
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
        # Pass 1 — collect per-detection data; only draw the highest-confidence door
        # ------------------------------------------------------------------
        det_data = []

        # Keep only the single highest-confidence detection
        if detections:
            detections = [max(detections, key=lambda d: d[4])]
        else:
            _ema_width = None
            _ema_depth = None

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

            # Smooth width and depth for display only; raw values kept for path logic
            _ema_width = ema_update(_ema_width, width_mm)
            _ema_depth = ema_update(_ema_depth, z_centre)
            display_width_mm = int((_ema_width if _ema_width is not None else width_mm) * 0.8)
            display_z_centre = int(_ema_depth) if _ema_depth is not None else z_centre

            det_data.append(dict(
                x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                label=label, color=color,
                z_centre=z_centre, z_left=z_left, z_right=z_right,  # raw — path logic
                width_mm=display_width_mm,          # smoothed — clearance card
                display_z_centre=display_z_centre,  # smoothed — distance card
                angle_deg=angle_deg,
            ))

            angle_str = f"{angle_deg:+.1f}°" if angle_deg is not None else "--"
            print(f"  {label} {int(conf*100)}%  x1={bx1:.1f} y1={by1:.1f} x2={bx2:.1f} y2={by2:.1f}  "
                  f"Z_left={z_left}mm  Z_centre={z_centre}mm  Z_right={z_right}mm  "
                  f"W={width_mm}mm ({width_mm/1000:.2f}m)  angle={angle_str}", flush=True)

            # Solid bounding box: white outline + coloured inner border
            cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # ------------------------------------------------------------------
        # Pass 2 — draw nav path for the primary open/semi door only;
        #          draw X/BLOCKED for every closed door
        # ------------------------------------------------------------------
        primary_target = select_primary_target(det_data)

        if primary_target is not None and primary_target["label"] in ("open", "semi"):
            draw_nav_path(frame,
                          primary_target["x1"], primary_target["y1"],
                          primary_target["x2"], primary_target["y2"],
                          primary_target["label"], primary_target["color"],
                          focal_px, primary_target["z_centre"], primary_target["angle_deg"])

        for d in det_data:
            if d["label"] == "closed":
                draw_nav_path(frame,
                              d["x1"], d["y1"], d["x2"], d["y2"],
                              d["label"], d["color"],
                              focal_px, d["z_centre"], d["angle_deg"])

        # ── Overlay tracked persons on camera frame ───────────────────────
        person_count = 0
        if PERSON_TRACKING and last_tracklets is not None:
            for t in last_tracklets.tracklets:
                if t.status == dai.Tracklet.TrackingStatus.LOST:
                    continue
                roi  = t.roi.denormalize(CAM_W, CAM_H)
                px1  = max(0, int(roi.topLeft().x))
                py1  = max(0, int(roi.topLeft().y))
                px2  = min(CAM_W - 1, int(roi.bottomRight().x))
                py2  = min(CAM_H - 1, int(roi.bottomRight().y))
                if t.status == dai.Tracklet.TrackingStatus.TRACKED:
                    person_count += 1
                col = PERSON_COLOR if t.status == dai.Tracklet.TrackingStatus.TRACKED else (80, 80, 80)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 0), 4)
                cv2.rectangle(frame, (px1, py1), (px2, py2), col, 2)
                cv2.putText(frame, f"ID {t.id}", (px1 + 6, py1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(frame, f"ID {t.id}", (px1 + 6, py1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        if PERSON_TRACKING:
            badge = f"Persons: {person_count}"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx = CAM_W - bw - 12
            by = 28
            cv2.rectangle(frame, (bx - 6, by - bh - 4), (bx + bw + 6, by + 4), (0, 0, 0), -1)
            cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_COLOR, 2)

        summary = build_status_summary(primary_target, w)
        display_frame = render_portrait_display(frame, summary, fps)
        last_display_frame = display_frame

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
