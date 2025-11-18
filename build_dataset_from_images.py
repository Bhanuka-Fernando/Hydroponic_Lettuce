import os, re, json, glob, math, argparse
import numpy as np, pandas as pd, cv2

# ---- CLI args ----
p = argparse.ArgumentParser()
p.add_argument("--json", required=True, help="Path to GroundTruth_All_388_Images.json")
p.add_argument("--rgb_dir", required=True, help="Path to RGB image folder")
p.add_argument("--depth_dir", required=True, help="Path to Depth image folder")
p.add_argument("--out", default="train_weight_dataset.csv", help="Output CSV")
args = p.parse_args()

json_path  = args.json
rgb_root   = args.rgb_dir
depth_root = args.depth_dir
out_csv    = args.out

if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON not found: {json_path}")
if not os.path.isdir(rgb_root):
    raise NotADirectoryError(f"RGB dir not found: {rgb_root}")
if not os.path.isdir(depth_root):
    raise NotADirectoryError(f"Depth dir not found: {depth_root}")

# ---- Load JSON + intrinsics ----
J = json.load(open(json_path,"r"))
cam = (J.get("General",{}) or {}).get("Camera",{}) or {}
fx = (cam.get("depth_int",{}) or {}).get("fx", cam.get("fx"))
fy = (cam.get("depth_int",{}) or {}).get("fy", cam.get("fy"))
depth_scale = cam.get("DepthScale", cam.get("depth_scale"))
if fx is None or fy is None or depth_scale is None:
    raise KeyError("Missing fx/fy/DepthScale in JSON (General.Camera).")

meas = J.get("Measurements") or {}
print(f"[info] meas entries: {len(meas)} | fx={fx} fy={fy} depth_scale={depth_scale}")

# ---- Index files by stem ----
def stem(p): 
    s = os.path.splitext(os.path.basename(p))[0]
    return s

rgb_files   = glob.glob(os.path.join(rgb_root,   "*.*"))
depth_files = glob.glob(os.path.join(depth_root, "*.*"))
rgb_index   = {stem(p): p for p in rgb_files}
depth_index = {stem(p): p for p in depth_files}
print(f"[info] found RGB files: {len(rgb_files)}, Depth files: {len(depth_files)}")

# ---- Segmentation & feature extraction ----
def segment_plant(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25,20,20), (95,255,255))  # adjust if too tight/loose
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((7,7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    return mask

def features_from(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth_raw is None:
        return None
    mask  = segment_plant(rgb)
    plant = mask > 0
    if plant.sum() < 100:
        return None

    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    Z = np.nanmedian(depth_m[plant])
    if not np.isfinite(Z) or Z <= 0:
        return None

    m_per_px_x = Z / float(fx)
    m_per_px_y = Z / float(fy)
    m2_per_px  = m_per_px_x * m_per_px_y
    A_cm2 = plant.sum() * m2_per_px * 1e4  # m^2 -> cm^2

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    D_cm = float("nan"); hull_area = 0.0
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        (cx,cy),(wpx,hpx),_ = cv2.minAreaRect(cnt)
        cm_per_px = (m2_per_px ** 0.5) * 100.0
        D_cm = max(wpx, hpx) * cm_per_px
        hull = cv2.convexHull(cnt); hull_area = cv2.contourArea(hull)

    mask_density = float(plant.sum()) / hull_area if hull_area > 0 else float("nan")
    invalid_depth_ratio = float(np.mean(~np.isfinite(depth_m[plant])))
    return dict(A_cm2=float(A_cm2), D_cm=float(D_cm),
                mask_density=mask_density, invalid_depth_ratio=invalid_depth_ratio)

# ---- File matching: prefer explicit filenames in JSON; else fuzzy match by key/id ----
def find_rgb_depth_for(key, rec):
    # try explicit fields in JSON (if present)
    for k in ("RGB","RGB_Image","RGBPath"):
        v = rec.get(k)
        if isinstance(v,str):
            p = v if os.path.isabs(v) else os.path.join(os.path.dirname(json_path), v)
            if os.path.exists(p): 
                rgb_p = p
                break
    else:
        rgb_p = None

    for k in ("Depth","Depth_Image","DepthPath"):
        v = rec.get(k)
        if isinstance(v,str):
            p = v if os.path.isabs(v) else os.path.join(os.path.dirname(json_path), v)
            if os.path.exists(p): 
                depth_p = p
                break
    else:
        depth_p = None

    # if not found, fuzzy match by stems using key and any numeric token inside key
    def search(index):
        cands = [key]
        m = re.search(r"(\d+)", key)
        if m: cands.append(m.group(1))
        # exact stem
        for c in cands:
            if c in index:
                return index[c]
        # substring
        for s, pth in index.items():
            for c in cands:
                if c and c in s:
                    return pth
        return None

    if rgb_p is None:
        rgb_p = search(rgb_index)
    if depth_p is None:
        depth_p = search(depth_index)
    return rgb_p, depth_p

# ---- Build rows ----
rows = []
unmatched = 0
for pid, rec in meas.items():
    rgb_p, depth_p = find_rgb_depth_for(pid, rec)
    if not rgb_p or not depth_p:
        unmatched += 1
        continue
    feats = features_from(rgb_p, depth_p)
    if feats is None:
        continue
    rows.append({
        "plant_id": pid,
        "variety": rec.get("Variety",""),
        "rgb_path": os.path.relpath(rgb_p, start=os.getcwd()),
        "depth_path": os.path.relpath(depth_p, start=os.getcwd()),
        "A_cm2": feats["A_cm2"],
        "D_cm": feats["D_cm"],
        "W_g": float(rec.get("FreshWeightShoot", float("nan"))),
        "mask_density": feats["mask_density"],
        "invalid_depth_ratio": feats["invalid_depth_ratio"],
    })

df = pd.DataFrame(rows).dropna(subset=["W_g","A_cm2","D_cm"])
df.to_csv(out_csv, index=False)
print(f"[done] Saved {out_csv} with {len(df)} rows (unmatched measurement entries: {unmatched})")
print(df.head(8).to_string(index=False))
