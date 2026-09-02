"""
make_figures.py
----------------
Builds every figure used in the report directly from the JSON/CSV outputs
written by traditional_cv.py and deep_learning.py. No numbers here are
invented -- everything is read back from the saved metrics files.
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def fig_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(history["epoch"], history["train_loss"], label="Training loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("CNN Loss vs. Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Training accuracy")
    axes[1].plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Validation accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("CNN Accuracy vs. Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_training_curves.png"), dpi=150)
    plt.close(fig)


def fig_confusion(cm, title, fname):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["OK", "Defect"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["OK", "Defect"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


def fig_bar_comparison(trad, dl):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    trad_vals = [trad[m] * 100 for m in metrics]
    dl_vals = [dl[m] * 100 for m in metrics]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, trad_vals, w, label="Traditional CV (OpenCV + SVM)", color="#3B82C4")
    b2 = ax.bar(x + w / 2, dl_vals, w, label="Deep Learning (from-scratch CNN)", color="#E8813A")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Traditional CV vs. Deep Learning — Test Set Performance")
    ax.legend(loc="upper right")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.text(0.02, 0.02,
            f"Inference latency — Traditional CV: {trad['inference_ms_per_image']:.2f} ms/img "
            f"| Deep Learning: {dl['inference_ms_per_image']:.2f} ms/img",
            transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_bar_comparison.png"), dpi=150)
    plt.close(fig)


def fig_sample_detections(csv_name, title, fname, n=6):
    with open(os.path.join(OUT_DIR, csv_name)) as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(7)
    pick = rng.choice(len(rows), size=n, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
    label_map = {"0": "OK", "1": "Defect"}
    for ax, i in zip(axes.flat, pick):
        r = rows[i]
        img = cv2.imread(os.path.join(IMAGES_DIR, r["filename"]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        correct = r["true"] == r["pred"]
        color = "green" if correct else "red"
        tag = "CORRECT" if correct else "MISCLASSIFIED"
        ax.set_title(f"True: {label_map[r['true']]} | Pred: {label_map[r['pred']]}\n{tag}",
                     color=color, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


def fig_dataset_samples():
    with open(os.path.join(DATA_DIR, "labels.csv")) as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(3)
    ok_rows = [r for r in rows if r["label"] == "OK"]
    def_rows = [r for r in rows if r["label"] == "DEFECT"]
    pick_ok = rng.choice(len(ok_rows), 3, replace=False)
    pick_def = rng.choice(len(def_rows), 3, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.4))
    for ax, i in zip(axes[0], pick_ok):
        img = cv2.imread(os.path.join(IMAGES_DIR, ok_rows[i]["filename"]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray"); ax.set_title("OK", fontsize=9); ax.axis("off")
    for ax, i in zip(axes[1], pick_def):
        r = def_rows[i]
        img = cv2.imread(os.path.join(IMAGES_DIR, r["filename"]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray"); ax.set_title(f"Defect ({r['defect_type']})", fontsize=9); ax.axis("off")
    fig.suptitle("Sample Synthetic Product Images (Top: OK, Bottom: Defect)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_dataset_samples.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    trad = load_json("traditional_metrics.json")
    dl = load_json("dl_metrics.json")
    history = load_json("dl_history.json")

    fig_dataset_samples()
    fig_training_curves(history)
    fig_confusion(trad["confusion_matrix"], "Traditional CV — Confusion Matrix (Test Set)", "fig_cm_traditional.png")
    fig_confusion(dl["confusion_matrix"], "Deep Learning — Confusion Matrix (Test Set)", "fig_cm_dl.png")
    fig_bar_comparison(trad, dl)
    fig_sample_detections("traditional_predictions.csv", "Sample Detections — Traditional CV", "fig_samples_traditional.png")
    fig_sample_detections("dl_predictions.csv", "Sample Detections — Deep Learning (CNN)", "fig_samples_dl.png")
    print("Figures written to", FIG_DIR)
    print(os.listdir(FIG_DIR))
