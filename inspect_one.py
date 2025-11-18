# inspect_one.py
import json, os, cv2
from segment_utils import segment_mask   # or paste the function here

CFG = json.load(open("config.json", "r"))
rgb_dir = CFG["rgb_dir"]

# ---- choose which RGB to inspect ----
TARGET_STEM = "RGB_114"   # change this (e.g. "RGB_105", "RGB_27", ...)

rgb_files = sorted([f for f in os.listdir(rgb_dir) if not f.startswith(".")])
if not rgb_files:
    raise SystemExit("No RGB files found.")

chosen = next((f for f in rgb_files if TARGET_STEM in f), None)
if chosen is None:
    raise SystemExit(f"No file matching {TARGET_STEM} in {rgb_dir}")

rgb_path = os.path.join(rgb_dir, chosen)
print("Using RGB:", rgb_path)

rgb = cv2.imread(rgb_path)
if rgb is None:
    raise SystemExit(f"Failed to read {rgb_path}")

mask = segment_mask(rgb)

overlay = rgb.copy()
overlay[mask > 0] = (0, 255, 0)

cv2.imwrite("debug_preview.png", overlay)
print("Saved debug_preview.png – check that almost the whole lettuce is green.")
