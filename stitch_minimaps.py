#!/usr/bin/env python3
"""
stitch_minimaps.py — Builds a global map from a sequence of overlapping minimaps.

Usage:
    python3 stitch_minimaps.py [input_dir] [output_file]

    input_dir   — directory with mm_*.png files  (default: minimaps/minimaps)
    output_file — output image path               (default: full_map.png)

The script:
  1. Loads minimaps in natural numeric order (mm_1, mm_2, …).
  2. Masks out blue and white cross markers so they don't affect
     alignment or appear on the final map.
  3. Detects translation between consecutive frames via template matching.
  4. Composites all frames onto a single canvas using weighted blending
     (cross / road pixels are excluded from blending).
  5. Removes residual sparse gray road dots with median-based filtering.
"""

import cv2
import numpy as np
import os
import re
import sys
import glob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def natural_sort_key(path):
    """mm_1, mm_2, … mm_10, … mm_100, …"""
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', os.path.basename(path))]


def create_cross_mask(img):
    """
    Binary mask (255 = ignore) for blue and white cross markers.
    Calibrated from actual pixel samples:
      blue  — HSV H≈110, S>150, V>100
      white — HSV S<30, V>200
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blue = cv2.inRange(hsv, np.array([100, 120, 80]),
                             np.array([125, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 200]),
                              np.array([180, 35, 255]))

    mask = cv2.bitwise_or(blue, white)
    # Dilate to cover anti-aliased edges of the markers
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    return mask


def create_road_mask(img):
    """
    Mask sparse gray road dots — small isolated gray pixels
    (low saturation, medium brightness) that disappear after
    morphological opening.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Candidate road pixels: gray, medium brightness
    candidates = ((hsv[:, :, 1] < 45) &
                  (gray > 90) & (gray < 195)).astype(np.uint8) * 255

    # Opening removes small features; subtract to keep only small ones
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    small = cv2.subtract(candidates, opened)
    return small


# ---------------------------------------------------------------------------
# Offset detection
# ---------------------------------------------------------------------------

def find_offset(img1, img2, mask1, mask2):
    """
    Compute (dx, dy) such that placing img2 at (pos1 + (dx, dy))
    aligns it with img1.  Uses normalised cross-correlation on the
    center 50 % of img1 as template.
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Neutralise masked pixels
    for g, m in ((g1, mask1), (g2, mask2)):
        valid = g[m == 0]
        g[m > 0] = np.mean(valid) if valid.size else 128

    h, w = g1.shape
    mx, my = w // 4, h // 4
    template = g1[my:h - my, mx:w - mx]

    result = cv2.matchTemplate(g2, template, cv2.TM_CCOEFF_NORMED)
    _, conf, _, loc = cv2.minMaxLoc(result)

    dx = mx - loc[0]
    dy = my - loc[1]
    return dx, dy, conf


# ---------------------------------------------------------------------------
# Post-processing: remove residual road dots
# ---------------------------------------------------------------------------

def remove_road_dots(img):
    """
    Replace small isolated gray dots (roads) with local median colour.
    """
    road = create_road_mask(img)
    if np.count_nonzero(road) == 0:
        return img

    # Inpaint the road pixels using surrounding area
    result = cv2.inpaint(img, road, inpaintRadius=3,
                         flags=cv2.INPAINT_TELEA)
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "minimaps/minimaps"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "full_map.png"

    # --- discover files ---
    files = sorted(
        glob.glob(os.path.join(input_dir, "mm_*.png")),
        key=natural_sort_key,
    )
    if not files:
        print(f"ERROR: no mm_*.png files found in '{input_dir}'")
        sys.exit(1)

    n = len(files)
    print(f"[1/5] Found {n} minimaps in '{input_dir}'")

    # --- load ---
    print("[2/5] Loading images & building masks …")
    images = []
    cross_masks = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"  WARNING: cannot read {f}, skipping")
            continue
        images.append(img)
        cross_masks.append(create_cross_mask(img))

    n = len(images)
    ih, iw = images[0].shape[:2]
    print(f"       {n} images loaded, each {iw}×{ih}")

    # --- compute pairwise offsets ---
    print("[3/5] Computing offsets …")
    positions = [(0.0, 0.0)]
    for i in range(n - 1):
        dx, dy, conf = find_offset(
            images[i], images[i + 1],
            cross_masks[i], cross_masks[i + 1],
        )
        px, py = positions[-1]
        positions.append((px + dx, py + dy))

        if conf < 0.40:
            print(f"  WARNING: low confidence {conf:.3f} at "
                  f"frame {i}→{i+1} (dx={dx}, dy={dy})")

    # normalise to non-negative integer coordinates
    min_x = min(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    int_positions = [
        (int(round(p[0] - min_x)), int(round(p[1] - min_y)))
        for p in positions
    ]

    canvas_w = max(p[0] for p in int_positions) + iw
    canvas_h = max(p[1] for p in int_positions) + ih
    print(f"       Canvas size: {canvas_w}×{canvas_h}")

    # --- composite ---
    print("[4/5] Compositing global map …")
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    for i in range(n):
        img = images[i]
        # Combined mask: crosses + road dots
        cmask = cv2.bitwise_or(cross_masks[i], create_road_mask(img))
        valid = (cmask == 0).astype(np.float64)

        px, py = int_positions[i]
        for c in range(3):
            canvas[py:py + ih, px:px + iw, c] += (
                img[:, :, c].astype(np.float64) * valid
            )
        weight[py:py + ih, px:px + iw] += valid

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"       placed {i + 1}/{n}")

    # weighted average
    weight = np.maximum(weight, 1.0)
    for c in range(3):
        canvas[:, :, c] /= weight

    result = np.clip(canvas, 0, 255).astype(np.uint8)

    # --- post-process: remove any residual road dots ---
    print("[5/5] Cleaning up residual road dots …")
    result = remove_road_dots(result)

    # --- save ---
    cv2.imwrite(output_file, result)
    print(f"Done → '{output_file}'  ({canvas_w}×{canvas_h})")


if __name__ == "__main__":
    main()
