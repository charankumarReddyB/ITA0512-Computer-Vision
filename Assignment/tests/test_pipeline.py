"""
Basic correctness tests for the defect-detection prototype.
Run with:  python -m pytest tests/ -v   (or plain: python tests/test_pipeline.py)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dataset as ds          # noqa: E402
import traditional_cv as tcv  # noqa: E402
import deep_learning as dl    # noqa: E402


def test_dataset_generation(tmp_path=None):
    rows = ds.generate(n_per_class=10, seed=1)
    assert len(rows) == 20
    labels = {r[1] for r in rows}
    assert labels == {"OK", "DEFECT"}


def test_traditional_feature_vector_shape():
    img, geom = tcv.preprocess.__globals__["cv2"], None  # sanity import check
    gray = np.zeros((64, 64), dtype=np.uint8)
    gray[20:44, 20:44] = 200
    enh, mask = tcv.preprocess(gray)
    feats = tcv.extract_features(gray, enh, mask)
    assert len(feats) == 8
    assert all(np.isfinite(f) for f in feats)


def test_cnn_forward_backward_shapes():
    net = dl.TinyCNN()
    x = np.random.rand(4, 1, dl.IMG, dl.IMG).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    logits = net.forward(x)
    assert logits.shape == (4, 2)
    loss, probs, dlogits = dl.softmax_ce_loss(logits, y)
    assert np.isfinite(loss)
    assert probs.shape == (4, 2)
    net.backward(dlogits, lr=1e-3, t=1)  # should not raise


def test_cnn_can_overfit_small_batch():
    """Regression guard: verifies backprop correctness by checking the
    network can memorise a tiny separable batch (loss must decrease)."""
    net = dl.TinyCNN()
    x = np.random.rand(8, 1, dl.IMG, dl.IMG).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    losses = []
    for t in range(1, 61):
        logits = net.forward(x)
        loss, probs, dlogits = dl.softmax_ce_loss(logits, y)
        net.backward(dlogits, lr=3e-3, t=t)
        losses.append(loss)
    assert losses[-1] < losses[0]


if __name__ == "__main__":
    test_dataset_generation()
    test_traditional_feature_vector_shape()
    test_cnn_forward_backward_shapes()
    test_cnn_can_overfit_small_batch()
    print("All tests passed.")
