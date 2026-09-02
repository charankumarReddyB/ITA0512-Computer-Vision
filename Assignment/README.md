# Defect Detection System Decision using Computer Vision

ITA05 – Computer Vision | Assignment: comparison of traditional computer
vision and deep learning approaches for automated industrial defect
detection, with a working prototype of both pipelines.

## Problem

A manufacturing company needs an automated computer-vision defect
detection system for a high-speed production line, targeting **>95%
detection accuracy** while remaining robust to variation in product
appearance, lighting, orientation, and manufacturing batch. This project
implements and quantitatively compares:

- **Approach A — Traditional CV:** OpenCV preprocessing (CLAHE, Otsu
  thresholding, morphology) + hand-engineered shape/texture features +
  an SVM classifier.
- **Approach B — Deep Learning:** a convolutional neural network,
  implemented from scratch in NumPy and trained end-to-end on raw pixels.

> **Note on Approach B:** the reference technology stack for production
> deep learning (TensorFlow/PyTorch + a pretrained MobileNet/EfficientNet
> backbone via transfer learning) requires internet access to fetch
> pretrained weights and framework packages. This development environment
> has no network access, so a small CNN was implemented and trained from
> scratch instead, as the closest executable alternative — see Section 4
> of the report for the implications of this substitution on the results.

## Dataset

No public dataset (e.g. MVTec-AD, NEU Surface Defect Database) could be
downloaded offline, so `src/dataset.py` procedurally generates a
360-image synthetic dataset of grayscale annular "product" images
(180 `OK`, 180 `DEFECT` with scratch/dent/chip defects), with randomised
lighting, orientation, and batch-size jitter to emulate the variation
named in the problem statement. Generation is seeded (`SEED=42`) for
reproducibility.

## Project Structure

```
proj/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py          # synthetic data generator
│   ├── traditional_cv.py   # Approach A: OpenCV + SVM
│   ├── deep_learning.py    # Approach B: from-scratch CNN
│   ├── make_figures.py     # builds all result figures from saved metrics
│   └── make_diagrams.py    # builds architecture/pipeline diagrams
├── data/
│   ├── images/              # generated product images
│   └── labels.csv
├── outputs/
│   ├── traditional_metrics.json
│   ├── dl_metrics.json
│   ├── dl_history.json
│   ├── *_predictions.csv
│   └── figures/             # all report figures (PNG)
├── tests/
│   └── test_pipeline.py
└── docs/
    └── Defect_Detection_Report.docx
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python3 src/dataset.py          # 1. generate the synthetic dataset
python3 src/traditional_cv.py   # 2. run Approach A, writes outputs/traditional_metrics.json
python3 src/deep_learning.py    # 3. run Approach B, writes outputs/dl_metrics.json
python3 src/make_figures.py     # 4. build result figures from the metrics above
python3 src/make_diagrams.py    # 5. build architecture diagrams
```

## Testing

```bash
python3 -m pytest tests/ -v
# or, without pytest:
python3 tests/test_pipeline.py
```

`tests/test_pipeline.py` checks: dataset generation produces the expected
class balance; the traditional-CV feature vector has the expected shape
and is numerically finite; the CNN's forward/backward pass run without
shape errors; and, as a backprop-correctness regression guard, that the
CNN's training loss decreases when memorising a small batch.

## Results (measured on this repository's synthetic test set, n = 108)

| Metric | Traditional CV (OpenCV+SVM) | Deep Learning (from-scratch CNN) |
|---|---|---|
| Accuracy | 80.6% | 65.7% |
| Precision | 83.7% | 66.0% |
| Recall | 75.9% | 64.8% |
| F1-score | 79.6% | 65.4% |
| Inference latency | ~0.4 ms/image | ~0.4 ms/image |
| Training data required | none (rule-based features) | 252 labelled images |

These are exact figures from the last run of this repository's fixed
seed; re-running the scripts above will reproduce them deterministically.
See the accompanying report (`docs/`) for full discussion, including why
the from-scratch CNN underperforms the classical pipeline here and what
that implies for the production recommendation.

## Troubleshooting

- `ModuleNotFoundError: cv2` → `pip install opencv-python`.
- Figures fail to save / blank plots → ensure `matplotlib` backend is
  non-interactive (`make_figures.py`/`make_diagrams.py` already set
  `matplotlib.use("Agg")`).
- Re-running `traditional_cv.py` / `deep_learning.py` requires
  `data/labels.csv` to exist — run `dataset.py` first.
- To reproduce results exactly, do not change `SEED=42` in `dataset.py`
  or the `random_state=42` train/test split in the two pipeline scripts.

## Acknowledgement of AI-Assisted Development

Claude (Anthropic) was used to help scaffold the synthetic dataset
generator, the from-scratch CNN implementation, figure generation
scripts, and this report, under the author's direction and review, per
the assignment's instruction to acknowledge AI-assisted tools used.
