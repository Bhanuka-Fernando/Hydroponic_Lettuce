import json, os, cv2, numpy as np

# ---- load config ----
CFG = json.load(open("config.json", "r"))
rgb_dir = CFG["rgb_dir"]

# ---- list RGB files ----
rgb_files = sorted([f for f in os.listdir(rgb_dir) if not f.startswith(".")])
if not rgb_files:
    raise SystemExit("No files in RGBImages. Check config.json paths.")

# ---- choose RGB_105.png ----
TARGET_STEM = "RGB_6"   # you can change this to test other images

# find a file whose name contains "RGB_105"
chosen = next((f for f in rgb_files if TARGET_STEM in f), None)
if chosen is None:
    raise SystemExit(f"No file matching {TARGET_STEM} found in {rgb_dir}")

rgb_path = os.path.join(rgb_dir, chosen)
print("Using RGB:", rgb_path)

rgb = cv2.imread(rgb_path)
if rgb is None:
    raise SystemExit(f"Failed to read {rgb_path}")

h, w = rgb.shape[:2]

def segment_mask(rgb):
    h, w = rgb.shape[:2]

    # widen vertical crop so it always covers the whole plant
    y1, y2 = int(0.25 * h), int(0.95 * h)   # was 0.35, 0.90
    x1, x2 = int(0.30 * w), int(0.85 * w)

    crop = rgb[y1:y2, x1:x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 1) colored pixels
    mask_sat = cv2.inRange(S, 60, 255)
    # 2) not pure white / not very dark
    mask_val = cv2.inRange(V, 30, 240)
    # 3) broad green/yellow band
    mask_hue = cv2.inRange(H, 20, 100)

    raw_mask = cv2.bitwise_and(mask_sat, mask_val)
    raw_mask = cv2.bitwise_and(raw_mask, mask_hue)

    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  np.ones((7, 7), np.uint8))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_mask = np.zeros_like(raw_mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(crop_mask, [biggest], -1, 255, thickness=-1)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = crop_mask

    return full_mask


mask = segment_mask(rgb)

overlay = rgb.copy()
overlay[mask > 0] = (0, 255, 0)

cv2.imwrite("debug_preview.png", overlay)
print("Saved debug_preview.png – check that almost the whole lettuce is green.")
