# predict_one.py

import os
import json
import cv2
import numpy as np
from segment_utils import segment_mask   # same segmentation as training

# ------------ CONFIG & PATHS ------------

CFG = json.load(open("config.json", "r"))
JSON_PATH = CFG["json_path"]
RGB_DIR   = CFG["rgb_dir"]
DEPTH_DIR = CFG["depth_dir"]
MIN_MASK  = int(CFG["min_mask_pixels"])

# Load camera intrinsics & depth scale
J = json.load(open(JSON_PATH, "r"))
cam = J.get("General", {}).get("Camera", {})

fx = cam.get("depth_int", {}).get("fx", cam.get("fx"))
fy = cam.get("depth_int", {}).get("fy", cam.get("fy"))
depth_scale = cam.get("DepthScale", cam.get("depth_scale"))

if fx is None or fy is None or depth_scale is None:
    raise SystemExit("fx/fy/DepthScale not found in JSON. Check General.Camera section.")
print(f"fx={fx}, fy={fy}, depth_scale={depth_scale}")

Measurements = J.get("Measurements", {})

# ------------ LOAD TRAINED MODEL PARAMS ------------

params = json.load(open("allometric_model.json", "r"))
alpha  = params["alpha"]
beta_A = params["beta_A"]
beta_D = params["beta_D"]

print(f"Loaded model: ln(W) = {alpha:.4f} + {beta_A:.4f} ln(A) + {beta_D:.4f} ln(D)")

# ------------ HOW TO CHOOSE IMAGE ------------

# Option A: manually choose RGB + Depth filenames
USE_MANUAL = True
MANUAL_RGB_NAME   = "RGB_6.png"      # change this to test other images
MANUAL_DEPTH_NAME = "Depth_6.png"    # change this to match

# Option B: choose by plant_id from JSON
TARGET_PID = "Image6"    # e.g. "Image6", "Image27", ...

# ------------ RESOLVE IMAGE + GT WEIGHT ------------

gt_W = None
gt_leaf_area = None
gt_diameter  = None
pid_for_print = None

if USE_MANUAL:
    rgb_path   = os.path.join(RGB_DIR, MANUAL_RGB_NAME)
    depth_path = os.path.join(DEPTH_DIR, MANUAL_DEPTH_NAME)
    pid_for_print = "(manual)"

    # try to find matching GT entry by filenames
    for pid, rec in Measurements.items():
        if rec.get("RGB_Image") == MANUAL_RGB_NAME and rec.get("Depth_Information") == MANUAL_DEPTH_NAME:
            gt_W          = rec.get("FreshWeightShoot", None)
            gt_leaf_area  = rec.get("LeafArea", None)
            gt_diameter   = rec.get("Diameter", None)
            pid_for_print = pid
            print(f"Matched GT record: plant_id={pid}")
            break
else:
    # choose by plant_id directly
    if TARGET_PID not in Measurements:
        raise SystemExit(f"plant_id {TARGET_PID} not in Measurements.")
    rec = Measurements[TARGET_PID]
    pid_for_print = TARGET_PID

    rgb_name   = rec["RGB_Image"]         # e.g. RGB_6.png
    depth_name = rec["Depth_Information"] # e.g. Depth_6.png

    rgb_path   = os.path.join(RGB_DIR, rgb_name)
    depth_path = os.path.join(DEPTH_DIR, depth_name)

    gt_W         = rec.get("FreshWeightShoot", None)
    gt_leaf_area = rec.get("LeafArea", None)
    gt_diameter  = rec.get("Diameter", None)

print(f"Using plant: {pid_for_print}")
print("RGB:  ", rgb_path)
print("DEPTH:", depth_path)

# ------------ LOAD IMAGES ------------

rgb       = cv2.imread(rgb_path)
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
if rgb is None or depth_raw is None:
    raise SystemExit("Failed to read RGB or Depth image.")

h_rgb, w_rgb = rgb.shape[:2]
h_d,   w_d   = depth_raw.shape[:2]

# ensure same size
if (h_rgb, w_rgb) != (h_d, w_d):
    print(f"[warn] size mismatch: rgb={h_rgb}x{w_rgb}, depth={h_d}x{w_d} -> resizing depth")
    depth_raw = cv2.resize(depth_raw, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)

# ------------ SEGMENT & COMPUTE A, D ------------

mask  = segment_mask(rgb)
plant = mask > 0
num_pixels = plant.sum()
print("Plant pixels:", num_pixels)

if num_pixels < MIN_MASK:
    raise SystemExit("Mask too small. Adjust thresholds or min_mask_pixels.")

# depth in meters
depth_m = depth_raw.astype(np.float32) * float(depth_scale)
Z = np.nanmedian(depth_m[plant])
print("Median depth Z (m):", Z)

if not np.isfinite(Z) or Z <= 0:
    raise SystemExit("Depth value invalid. Check DepthScale or depth image.")

m_per_px_x = Z / float(fx)
m_per_px_y = Z / float(fy)
m2_per_px  = m_per_px_x * m_per_px_y

A_m2  = num_pixels * m2_per_px
A_cm2 = A_m2 * 1e4

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    raise SystemExit("No contours on mask.")

cnt = max(cnts, key=cv2.contourArea)
(cx, cy), (wpx, hpx), angle = cv2.minAreaRect(cnt)

cm_per_px = (m2_per_px ** 0.5) * 100.0
D_cm = max(wpx, hpx) * cm_per_px

print(f"Estimated area A_cm2      = {A_cm2:.2f} cm^2")
print(f"Estimated Feret D_cm      = {D_cm:.2f} cm")

# ------------ PREDICT WEIGHT USING TRAINED MODEL ------------

logA = np.log(A_cm2)
logD = np.log(D_cm)

logW_pred = alpha + beta_A * logA + beta_D * logD
W_pred_g  = float(np.exp(logW_pred))

print(f"Predicted weight W_pred   = {W_pred_g:.2f} g")

if gt_W is not None:
    print(f"Ground-truth weight W_gt  = {gt_W:.2f} g")
if gt_leaf_area is not None:
    print(f"GT LeafArea               = {gt_leaf_area:.2f} cm^2")
if gt_diameter is not None:
    print(f"GT Diameter               = {gt_diameter:.2f} cm")

# ------------ SAVE DEBUG IMAGE ------------

dbg = rgb.copy()
box = cv2.boxPoints(((cx, cy), (wpx, hpx), angle)).astype(int)
cv2.drawContours(dbg, [box], 0, (255, 0, 0), 2)

text1 = f"A={A_cm2:.1f}cm^2 D={D_cm:.1f}cm"
text2 = f"W_pred={W_pred_g:.1f}g"
cv2.putText(dbg, text1, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.putText(dbg, text2, (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

out_name = "debug_predict_one.png"
cv2.imwrite(out_name, dbg)
print(f"Saved {out_name} with box + A/D + W_pred.")
