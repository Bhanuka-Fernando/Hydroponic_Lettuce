# test_one_metrics.py

import os, json, cv2, numpy as np
from segment_utils import segment_mask   # or paste function here

# ------------ CONFIG ------------

CFG = json.load(open("config.json", "r"))
JSON_PATH = CFG["json_path"]
RGB_DIR   = CFG["rgb_dir"]
DEPTH_DIR = CFG["depth_dir"]
MIN_MASK  = int(CFG["min_mask_pixels"])

J = json.load(open(JSON_PATH, "r"))
cam = J.get("General", {}).get("Camera", {})

fx = cam.get("depth_int", {}).get("fx", cam.get("fx"))
fy = cam.get("depth_int", {}).get("fy", cam.get("fy"))
depth_scale = cam.get("DepthScale", cam.get("depth_scale"))

if fx is None or fy is None or depth_scale is None:
    raise SystemExit("fx/fy/DepthScale not found in JSON. Check General.Camera section.")
print(f"fx={fx}, fy={fy}, depth_scale={depth_scale}")

Measurements = J.get("Measurements", {})
if not Measurements:
    raise SystemExit("No Measurements found in JSON.")

# ------------ choose plant ------------

TARGET_PID = "Image27"   # change to test another plant id

if TARGET_PID in Measurements:
    pid = TARGET_PID
else:
    pid = next(iter(Measurements.keys()))

rec = Measurements[pid]
print("Using plant_id:", pid)

rgb_name   = rec["RGB_Image"]         # e.g. "RGB_27.png"
depth_name = rec["Depth_Information"] # e.g. "Depth_27.png"

rgb_path   = os.path.join(RGB_DIR, rgb_name)
depth_path = os.path.join(DEPTH_DIR, depth_name)

print("RGB:", rgb_path)
print("DEPTH:", depth_path)

rgb       = cv2.imread(rgb_path)
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
if rgb is None or depth_raw is None:
    raise SystemExit("Failed to read RGB or Depth.")

gt_leaf_area = rec.get("LeafArea")
gt_diameter  = rec.get("Diameter")
print(f"GT LeafArea: {gt_leaf_area} cm^2, GT Diameter: {gt_diameter} cm")

# ------------ compute A & D ------------

mask  = segment_mask(rgb)
plant = mask > 0
num_pixels = plant.sum()
print("Plant pixels:", num_pixels)

if num_pixels < MIN_MASK:
    raise SystemExit("Mask too small. Adjust thresholds or min_mask_pixels.")

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

print(f"Estimated area: {A_cm2:.2f} cm^2")
print(f"Estimated Feret diameter: {D_cm:.2f} cm")

# simple error print
if gt_leaf_area is not None:
    print(f"Area ratio (our/GT): {A_cm2/gt_leaf_area:.2f}")
if gt_diameter is not None:
    print(f"Diameter diff: {D_cm-gt_diameter:.2f} cm")

# debug image
dbg = rgb.copy()
box = cv2.boxPoints(((cx, cy), (wpx, hpx), angle)).astype(int)
cv2.drawContours(dbg, [box], 0, (255, 0, 0), 2)
cv2.putText(dbg, f"A={A_cm2:.1f}cm^2 D={D_cm:.1f}cm",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.imwrite("debug_one_metrics.png", dbg)
print("Saved debug_one_metrics.png with rectangle and A/D text.")
