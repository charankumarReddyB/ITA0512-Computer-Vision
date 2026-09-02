"""
traditional_cv.py
------------------
Approach A: classical computer-vision defect detection.

Pipeline: grayscale -> Gaussian denoise -> CLAHE illumination correction
-> Otsu thresholding -> contour extraction -> morphological cleanup ->
shape + texture feature computation -> SVM classification.

No learned features: every stage is hand-engineered, which is the
defining trait (and limitation) of Approach A.
"""
import os
import json
import time
import csv
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def preprocess(gray):
    """Denoise + illumination correction (CLAHE) -> Otsu mask -> morphology."""
    den = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(den)
    _, mask = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return enh, mask


def extract_features(gray, enh, mask):
    """Shape features (contour) + texture features (GLCM-style stats, LBP-lite)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        largest = contours[int(np.argmax(areas))]
        area = cv2.contourArea(largest)
        perim = cv2.arcLength(largest, True)
        circularity = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0
        n_components = len(contours)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
    else:
        area = perim = circularity = solidity = 0
        n_components = 0

    # texture: local intensity variance inside the masked region (proxy for
    # scratches/dents disturbing an otherwise uniform surface)
    region_vals = enh[mask > 0]
    tex_std = float(np.std(region_vals)) if region_vals.size else 0.0
    tex_mean = float(np.mean(region_vals)) if region_vals.size else 0.0

    # edge density via Canny (gradient/edge feature)
    edges = cv2.Canny(enh, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size

    return [area, perim, circularity, solidity, n_components,
            tex_std, tex_mean, edge_density]


def run():
    with open(os.path.join(DATA_DIR, "labels.csv")) as f:
        rows = list(csv.DictReader(f))

    X, y, fnames = [], [], []
    t_feat_start = time.time()
    for r in rows:
        gray = cv2.imread(os.path.join(IMAGES_DIR, r["filename"]), cv2.IMREAD_GRAYSCALE)
        enh, mask = preprocess(gray)
        feats = extract_features(gray, enh, mask)
        X.append(feats)
        y.append(1 if r["label"] == "DEFECT" else 0)
        fnames.append(r["filename"])
    feat_time_per_img_ms = (time.time() - t_feat_start) / len(rows) * 1000

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X, y, fnames, test_size=0.30, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    clf.fit(X_train_s, y_train)

    t0 = time.time()
    y_pred = clf.predict(X_test_s)
    infer_ms_per_img = (time.time() - t0) / len(X_test_s) * 1000 + feat_time_per_img_ms

    metrics = {
        "approach": "Traditional CV (OpenCV feature engineering + SVM)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "inference_ms_per_image": infer_ms_per_img,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "traditional_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # save test predictions for the report's sample-detection figure
    with open(os.path.join(OUT_DIR, "traditional_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true", "pred"])
        for fn, t, p in zip(f_test, y_test, y_pred):
            w.writerow([fn, int(t), int(p)])

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
