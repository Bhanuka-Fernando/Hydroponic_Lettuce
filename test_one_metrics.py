import os, json, cv2, numpy as np

# ------------- CONFIG & GLOBALS -------------

CFG = json.load(open("config.json", "r"))
JSON_PATH = CFG["json_path"]
RGB_DIR   = CFG["rgb_dir"]
DEPTH_DIR = CFG["depth_dir"]
HSV_L     = tuple(CFG["hsv_lower"])
HSV_U     = tuple(CFG["hsv_upper"])
MIN_MASK  = int(CFG["min_mask_pixels"])

# ---- load intrinsics & depth scale ----
J = json.load(open(JSON_PATH, "r"))
cam = J.get("General", {}).get("Camera", {})

fx = cam.get("depth_int", {}).get("fx", cam.get("fx"))
fy = cam.get("depth_int", {}).get("fy", cam.get("fy"))
depth_scale = cam.get("DepthScale", cam.get("depth_scale"))

if fx is None or fy is None or depth_scale is None:
    raise SystemExit("fx/fy/DepthScale not found in JSON. Check General.Camera section.")

print(f"fx={fx}, fy={fy}, depth_scale={depth_scale}")

# ------------- SEGMENTATION -------------

def segment_mask(rgb):
    h, w = rgb.shape[:2]

    # wider crop so it always covers the whole plant
    y1, y2 = int(0.25 * h), int(0.95 * h)
    x1, x2 = int(0.30 * w), int(0.85 * w)

    crop = rgb[y1:y2, x1:x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # colored pixels
    mask_sat = cv2.inRange(S, 60, 255)
    # not pure white / not very dark
    mask_val = cv2.inRange(V, 30, 240)
    # broad green/yellow band
    mask_hue = cv2.inRange(H, 20, 100)

    raw_mask = cv2.bitwise_and(mask_sat, mask_val)
    raw_mask = cv2.bitwise_and(raw_mask, mask_hue)

    # clean up
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  np.ones((7, 7), np.uint8))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # largest blob in crop = plant
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_mask = np.zeros_like(raw_mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(crop_mask, [biggest], -1, 255, thickness=-1)

    # expand back to full image
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = crop_mask

    return full_mask

# ------------- INDEX FILES BY STEM -------------

def stem(p):
    return os.path.splitext(os.path.basename(p))[0]

rgb_paths   = [os.path.join(RGB_DIR, f) for f in os.listdir(RGB_DIR) if not f.startswith(".")]
depth_paths = [os.path.join(DEPTH_DIR, f) for f in os.listdir(DEPTH_DIR) if not f.startswith(".")]

rgb_idx   = {stem(p): p for p in rgb_paths}
depth_idx = {stem(p): p for p in depth_paths}

# ------------- PICK ONE MEASUREMENT -------------

Measurements = J.get("Measurements", {})
if not Measurements:
    raise SystemExit("No Measurements found in JSON.")

# choose a specific plant (e.g. Image27) – change this if you want another
TARGET_PID = "Image27"
if TARGET_PID in Measurements:
    pid = TARGET_PID
else:
    # fallback: first entry
    pid = next(iter(Measurements.keys()))

rec = Measurements[pid]
print("Using plant_id:", pid)

# get filenames from JSON (e.g. RGB_27.png, Depth_27.png)
rgb_stem   = os.path.splitext(rec["RGB_Image"])[0]
depth_stem = os.path.splitext(rec["Depth_Information"])[0]

rgb_path   = rgb_idx.get(rgb_stem)
depth_path = depth_idx.get(depth_stem)

if not rgb_path or not depth_path:
    raise SystemExit(f"Could not find files for {pid}: {rgb_stem}, {depth_stem}")

print("RGB:", rgb_path)
print("DEPTH:", depth_path)

rgb = cv2.imread(rgb_path)
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if rgb is None or depth_raw is None:
    raise SystemExit("Failed to read RGB or Depth.")

# print ground-truth values for comparison
gt_leaf_area = rec.get("LeafArea", None)
gt_diameter  = rec.get("Diameter", None)
print(f"GT LeafArea: {gt_leaf_area} cm^2, GT Diameter: {gt_diameter} cm")

# ------------- COMPUTE A & D -------------

mask = segment_mask(rgb)
plant = mask > 0
num_pixels = plant.sum()
print("Plant pixels:", num_pixels)

if num_pixels < MIN_MASK:
    raise SystemExit("Mask too small. Adjust HSV or min_mask_pixels.")

depth_m = depth_raw.astype(np.float32) * float(depth_scale)  # meters
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
    raise SystemExit("No contours found on mask.")

cnt = max(cnts, key=cv2.contourArea)
(cx, cy), (wpx, hpx), angle = cv2.minAreaRect(cnt)

cm_per_px = (m2_per_px ** 0.5) * 100.0
D_cm = max(wpx, hpx) * cm_per_px

print(f"Estimated area: {A_cm2:.2f} cm^2")
print(f"Estimated Feret diameter: {D_cm:.2f} cm")

dbg = rgb.copy()
box = cv2.boxPoints(((cx, cy), (wpx, hpx), angle)).astype(int)
cv2.drawContours(dbg, [box], 0, (255, 0, 0), 2)
cv2.putText(
    dbg,
    f"A={A_cm2:.1f}cm^2 D={D_cm:.1f}cm",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2,
)

cv2.imwrite("debug_one_metrics.png", dbg)
print("Saved debug_one_metrics.png with rectangle and A/D text.")
