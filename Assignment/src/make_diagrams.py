"""
make_diagrams.py
-----------------
Original architecture / pipeline diagrams for the report, drawn with
matplotlib boxes+arrows (deliberately plain/technical rather than the
icon-based style used in the reference report, to keep this visually and
structurally distinct while conveying the same engineering content).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#EAF1FB", ec="#2A5C9A"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                        fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                         color="#333333", lw=1.2)
    ax.add_patch(a)


def pipeline_diagram():
    stages = ["Image\nAcquisition", "Preprocessing\n(denoise, CLAHE)",
              "Branch:\nApproach A / B", "Feature Extraction /\nRepresentation Learning",
              "Classification /\nDetection", "Defect\nLocalization",
              "Performance\nEvaluation", "Quality\nInspection Report"]
    fig, ax = plt.subplots(figsize=(11, 2.6))
    n = len(stages)
    w, h, gap = 1.15, 0.9, 0.35
    x = 0.1
    for s in stages:
        box(ax, x, 0.3, w, h, s)
        if x + w + gap < n * (w + gap):
            arrow(ax, x + w, 0.75, x + w + gap, 0.75)
        x += w + gap
    ax.set_xlim(0, x)
    ax.set_ylim(0, 1.5)
    ax.axis("off")
    ax.set_title("Figure 1. End-to-End Defect Detection Pipeline", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_pipeline.png"), dpi=150)
    plt.close(fig)


def traditional_diagram():
    stages = ["Grayscale\nInput", "Gaussian\nDenoise", "CLAHE\nIllum. Correction",
              "Otsu\nThreshold", "Morphological\nCleanup", "Contour +\nTexture Features",
              "SVM\nClassifier", "OK / Defect"]
    fig, ax = plt.subplots(figsize=(11, 2.6))
    w, h, gap = 1.15, 0.9, 0.32
    x = 0.1
    for s in stages:
        box(ax, x, 0.3, w, h, s, fc="#EAF6EE", ec="#2E7D4F")
        if x + w + gap < len(stages) * (w + gap):
            arrow(ax, x + w, 0.75, x + w + gap, 0.75)
        x += w + gap
    ax.set_xlim(0, x); ax.set_ylim(0, 1.5); ax.axis("off")
    ax.set_title("Figure 2. Approach A — Traditional Computer Vision Pipeline", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_traditional_pipeline.png"), dpi=150)
    plt.close(fig)


def cnn_diagram():
    fig, ax = plt.subplots(figsize=(11, 2.8))
    layers = ["Input\n32x32x1", "Conv 3x3 (8)\n+ ReLU", "MaxPool\n2x2",
              "Conv 3x3 (16)\n+ ReLU", "MaxPool\n2x2", "Flatten", "Dense (64)\n+ ReLU",
              "Dense (2)\n+ Softmax", "OK / Defect"]
    w, h, gap = 1.1, 0.9, 0.28
    x = 0.1
    for s in layers:
        box(ax, x, 0.3, w, h, s, fc="#FDF0E3", ec="#C1671C")
        if x + w + gap < len(layers) * (w + gap):
            arrow(ax, x + w, 0.75, x + w + gap, 0.75)
        x += w + gap
    ax.set_xlim(0, x); ax.set_ylim(0, 1.5); ax.axis("off")
    ax.set_title("Figure 3. Approach B — CNN Architecture (Trained From Scratch)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_cnn_architecture.png"), dpi=150)
    plt.close(fig)


def module_interaction_diagram():
    fig, ax = plt.subplots(figsize=(9, 5))
    box(ax, 3.4, 4.1, 2.2, 0.7, "dataset.py\n(Synthetic Data Generator)")
    box(ax, 0.4, 2.7, 2.6, 0.7, "traditional_cv.py\n(Approach A)")
    box(ax, 6.0, 2.7, 2.6, 0.7, "deep_learning.py\n(Approach B)")
    box(ax, 3.4, 1.3, 2.2, 0.7, "make_figures.py\n(Evaluation + Plots)")
    box(ax, 3.4, 0.0, 2.2, 0.7, "outputs/\n(metrics, figures, predictions)")

    arrow(ax, 4.0, 4.1, 1.7, 3.4)
    arrow(ax, 5.4, 4.1, 7.3, 3.4)
    arrow(ax, 1.7, 2.7, 4.0, 2.0)
    arrow(ax, 7.3, 2.7, 4.8, 2.0)
    arrow(ax, 4.5, 1.3, 4.5, 0.7)

    ax.set_xlim(0, 9); ax.set_ylim(-0.3, 5.2); ax.axis("off")
    ax.set_title("Figure 4. Module Interaction Diagram", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_module_interaction.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    pipeline_diagram()
    traditional_diagram()
    cnn_diagram()
    module_interaction_diagram()
    print("done")
