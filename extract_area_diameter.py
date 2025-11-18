# extract_area_diameter.py

import os, re, json, glob
import numpy as np, pandas as pd, cv2
from tqdm import tqdm
from segment_utils import segment_mask   # or paste same function here

# ------------ CONFIG ------------

CFG = json.load(open("config.json", "r"))
JSON_PATH = CFG["json_path"]
RGB_DIR   = CFG["rgb_dir"]
DEPTH_DIR = CFG["depth_dir"]
OUT_CSV   = CFG["out_csv"]
DEBUG_DIR = CFG["debug_dir"]
MIN_MASK  = int(CFG["min_mask_pixels"])

os.makedirs(DEBUG_DIR, exist_ok=True)

J = json.load(open(JSON_PATH, "r"))
cam = J.get("General", {}).get("Camera", {})

fx = cam.get("depth_int", {}).get("fx", cam.get("fx"))
fy = cam.get("depth_int", {}).get("fy", cam.get("fy"))
depth_scale = cam.get("DepthScale", cam.get("depth_scale"))
if fx is None or fy is None or depth_scale is None:
    raise KeyError("Missing fx/fy/DepthScale in JSON under General.Camera")

Measurements = J.get("Measurements", {})
print(f"[info] intrinsics: fx={fx}, fy={fy}, depth_scale={depth_scale}")
print(f"[info] measurements entries: {len(Measurements)}")

# ------------ helpers ------------

def stem(p):
    return os.path.splitext(os.path.basename(p))[0]

rgb_paths   = glob.glob(os.path.join(RGB_DIR, "*.*"))
depth_paths = glob.glob(os.path.join(DEPTH_DIR, "*.*"))
rgb_idx   = {stem(p): p for p in rgb_paths}
depth_idx = {stem(p): p for p in depth_paths}

def find_match(idx, filename):
    # remove extension & use just stem
    s = os.path.splitext(filename)[0]
    if s in idx:
        return idx[s]
    # fallback: substring match
    for k, p in idx.items():
        if s in k:
            return p
    return None

def area_diameter_from(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth_raw is None:
        return None

    mask  = segment_mask(rgb)
    plant = mask > 0
    if plant.sum() < MIN_MASK:
        return None

    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    Z = np.nanmedian(depth_m[plant])
    if not np.isfinite(Z) or Z <= 0:
        return None

    m_per_px_x = Z / float(fx)
    m_per_px_y = Z / float(fy)
    m2_per_px  = m_per_px_x * m_per_px_y
    A_cm2 = float(plant.sum() * m2_per_px * 1e4)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    D_cm = float("nan"); hull_area = 0.0
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        (cx, cy), (wpx, hpx), angle = cv2.minAreaRect(cnt)
        cm_per_px = (m2_per_px ** 0.5) * 100.0
        D_cm = float(max(wpx, hpx) * cm_per_px)
        hull = cv2.convexHull(cnt); hull_area = float(cv2.contourArea(hull))

        # save debug rectangle
        dbg = rgb.copy()
        box = cv2.boxPoints(((cx, cy), (wpx, hpx), angle)).astype(int)
        cv2.drawContours(dbg, [box], 0, (255, 0, 0), 2)
        cv2.putText(dbg, f"A={A_cm2:.1f}cm^2 D={D_cm:.1f}cm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(DEBUG_DIR, os.path.basename(rgb_path)), dbg)

    mask_density = float(plant.sum()) / hull_area if hull_area > 0 else float("nan")
    invalid_depth_ratio = float(np.mean(~np.isfinite(depth_m[plant])))

    return dict(A_cm2=A_cm2, D_cm=D_cm,
                mask_density=mask_density,
                invalid_depth_ratio=invalid_depth_ratio)

# ------------ main loop ------------

rows = []
skipped = 0

for pid, rec in tqdm(Measurements.items(), desc="Processing plants"):
    rgb_path   = find_match(rgb_idx,   rec["RGB_Image"])
    depth_path = find_match(depth_idx, rec["Depth_Information"])

    if not rgb_path or not depth_path:
        skipped += 1
        continue

    feats = area_diameter_from(rgb_path, depth_path)
    if feats is None:
        skipped += 1
        continue

    rows.append({
        "plant_id": pid,
        "variety": rec.get("Variety", ""),
        "rgb_path": os.path.relpath(rgb_path),
        "depth_path": os.path.relpath(depth_path),
        **feats,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"[done] wrote {OUT_CSV} with {len(df)} rows, skipped={skipped}")
print(df.head().to_string(index=False))
