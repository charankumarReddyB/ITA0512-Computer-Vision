"""
dataset.py
----------
Generates a synthetic industrial "product" image dataset for the defect
detection experiment.

Why synthetic data: the assignment assumes a public industrial dataset
(e.g. MVTec-AD / NEU Surface Defect). This offline environment has no
network access, so a real dataset cannot be downloaded. Instead we
procedurally generate grayscale "washer/gear"-style product images
(a bright annulus on a dark background, matching the visual style of
common metal-part inspection datasets), with two classes:

  OK      - clean annulus, only sensor noise + mild lighting gradient
  DEFECT  - annulus with an added surface scratch, dent, or dark blob

Randomised lighting, rotation, and per-batch appearance jitter are
injected so the traditional-CV and deep-learning pipelines are tested
under the same "appearance / lighting / orientation / batch variation"
conditions named in the problem statement.

All images and labels are written to data/ and are regenerated
deterministically from SEED, so results are reproducible.
"""
import os
import csv
import numpy as np
import cv2

SEED = 42
IMG_SIZE = 64
N_PER_CLASS = 180  # 360 images total -> 70/15/15 split
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def _base_product(rng, size, batch_radius_jitter, lighting_bias):
    """Draw a clean annular 'product' (e.g. washer/gear blank)."""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size / 2 + rng.uniform(-2, 2), size / 2 + rng.uniform(-2, 2)
    outer_r = size * 0.38 * (1 + batch_radius_jitter)
    inner_r = outer_r * rng.uniform(0.30, 0.42)

    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = (dist <= outer_r) & (dist >= inner_r)
    img[ring] = 0.55 + lighting_bias

    # simulated ambient/inspection-station lighting gradient
    grad = (xx / size) * rng.uniform(-0.15, 0.15) + (yy / size) * rng.uniform(-0.1, 0.1)
    img += grad
    img = np.clip(img, 0, 1)
    return img, (cx, cy, outer_r, inner_r)


def _add_defect(img, rng, geom):
    """Add one of: scratch line, dark dent blob, or missing-edge chip."""
    cx, cy, outer_r, inner_r = geom
    size = img.shape[0]
    kind = rng.choice(["scratch", "dent", "chip"])

    if kind == "scratch":
        ang = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(inner_r, outer_r)
        length = rng.uniform(6, 14)
        x0 = int(cx + r0 * np.cos(ang))
        y0 = int(cy + r0 * np.sin(ang))
        x1 = int(x0 + length * np.cos(ang + rng.uniform(-0.3, 0.3)))
        y1 = int(y0 + length * np.sin(ang + rng.uniform(-0.3, 0.3)))
        cv2.line(img, (x0, y0), (x1, y1), color=float(0.15), thickness=1)
    elif kind == "dent":
        ang = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(inner_r, outer_r)
        x0 = int(cx + r0 * np.cos(ang))
        y0 = int(cy + r0 * np.sin(ang))
        rad = int(rng.uniform(2, 4))
        cv2.circle(img, (x0, y0), rad, color=float(0.10), thickness=-1)
    else:  # chip: erase part of the outer edge
        ang = rng.uniform(0, 2 * np.pi)
        x0 = int(cx + outer_r * np.cos(ang))
        y0 = int(cy + outer_r * np.sin(ang))
        rad = int(rng.uniform(3, 6))
        cv2.circle(img, (x0, y0), rad, color=float(0.0), thickness=-1)
    return img, kind


def _add_sensor_noise(img, rng, sigma=0.035):
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1)


def generate(n_per_class=N_PER_CLASS, size=IMG_SIZE, seed=SEED):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    idx = 0
    for label, n in (("OK", n_per_class), ("DEFECT", n_per_class)):
        for _ in range(n):
            batch_jitter = rng.uniform(-0.06, 0.06)   # batch-to-batch size variation
            lighting_bias = rng.uniform(-0.08, 0.08)  # lighting variation
            img, geom = _base_product(rng, size, batch_jitter, lighting_bias)
            defect_kind = "none"
            if label == "DEFECT":
                img, defect_kind = _add_defect(img, rng, geom)
            img = _add_sensor_noise(img, rng)
            # orientation variation
            angle = rng.uniform(0, 360)
            M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (size, size), borderValue=0)

            fname = f"{label}_{idx:04d}.png"
            cv2.imwrite(os.path.join(IMAGES_DIR, fname), (img * 255).astype(np.uint8))
            rows.append([fname, label, defect_kind])
            idx += 1

    with open(os.path.join(DATA_DIR, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "defect_type"])
        w.writerows(rows)

    print(f"Generated {len(rows)} images -> {IMAGES_DIR}")
    return rows


if __name__ == "__main__":
    generate()
