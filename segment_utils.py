# segment_utils.py  (optional helper file – or just copy this function into each script)

import cv2
import numpy as np

def segment_mask(rgb):
    """
    Segment the lettuce canopy on the tray.
    Returns a full-sized 0/255 mask (uint8).
    """
    h, w = rgb.shape[:2]

    # wide crop that covers tray area
    y1, y2 = int(0.25 * h), int(0.95 * h)
    x1, x2 = int(0.30 * w), int(0.85 * w)
    crop = rgb[y1:y2, x1:x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 1) colored pixels (ignore grey/white)
    mask_sat = cv2.inRange(S, 60, 255)
    # 2) not too bright / not too dark
    mask_val = cv2.inRange(V, 30, 240)
    # 3) broad green/yellow hue band
    mask_hue = cv2.inRange(H, 20, 100)

    raw = cv2.bitwise_and(mask_sat, mask_val)
    raw = cv2.bitwise_and(raw, mask_hue)

    # clean up
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  np.ones((7, 7), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # largest blob in crop = plant
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_mask = np.zeros_like(raw)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(crop_mask, [biggest], -1, 255, thickness=-1)

    # expand back to full image
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = crop_mask

    return full_mask
