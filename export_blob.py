#!/usr/bin/env python3
"""
Converts best.pt → best.blob for use with the OAK camera.
Run once: python export_blob.py
"""
from pathlib import Path
from ultralytics import YOLO
import blobconverter

PT_PATH = Path(__file__).parent / "best.pt"
IMGSZ   = 640
SHAVES  = 6   # 6 is standard for OAK-D; reduce to 5 if you get memory errors

print("Step 1: Exporting to ONNX (opset 12)...")
model = YOLO(str(PT_PATH))
model.export(format="onnx", imgsz=IMGSZ, opset=12)
# produces best.onnx next to best.pt

onnx_path = PT_PATH.with_suffix(".onnx")
if not onnx_path.exists():
    raise FileNotFoundError(f"Export failed — {onnx_path} not found")

print("Step 2: Converting ONNX → .blob...")
blob_path = blobconverter.from_onnx(
    model=str(onnx_path),
    shaves=SHAVES,
    data_type="FP16",
    optimizer_params=[
        "--scale_values=images[255,255,255]",  # normalise uint8 [0-255] → [0-1]
        "--reverse_input_channels",           # BGR (OAK) → RGB (YOLOv8)
    ],
    output_dir=str(PT_PATH.parent),
)

print(f"\nDone! Blob saved to:\n  {blob_path}")
print(f"\nRun detection with:\n  python detect_depth.py \"{blob_path}\"")
