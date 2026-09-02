"""
deep_learning.py
-----------------
Approach B: deep learning defect detection.

Environment constraint: this offline container has no network access, so
TensorFlow/PyTorch (and any pretrained MobileNet/EfficientNet weights)
cannot be installed or downloaded. Per the assignment's own instruction
("if a component cannot realistically be implemented in the available
environment, explicitly state that and provide the closest executable
alternative"), the closest faithful executable alternative is used here:
a small convolutional neural network implemented from scratch in NumPy
(im2col convolutions, backprop, Adam optimizer), trained end-to-end on
the same synthetic dataset and the same train/test split as Approach A.

This keeps the defining property of "deep learning" that the report
argues for -- LEARNED hierarchical features from raw pixels rather than
hand-engineered ones -- while being honestly labelled as a compact CNN
rather than a transfer-learning MobileNet/EfficientNet model that would
require internet access to obtain pretrained weights.
"""
import os
import json
import time
import csv
import numpy as np
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
IMG = 32  # downsampled for tractable from-scratch training
rng = np.random.default_rng(42)


# ---------------------------------------------------------------- im2col ops
def im2col(x, kh, kw, stride=1, pad=0):
    n, c, h, w = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (x.shape[2] - kh) // stride + 1
    ow = (x.shape[3] - kw) // stride + 1
    cols = np.zeros((n, c, kh, kw, oh, ow), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * oh
        for j in range(kw):
            j_max = j + stride * ow
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, -1)
    return cols, oh, ow


class Conv2D:
    def __init__(self, in_c, out_c, k=3, stride=1, pad=1):
        scale = np.sqrt(2.0 / (in_c * k * k))
        self.W = rng.normal(0, scale, (out_c, in_c, k, k)).astype(np.float32)
        self.b = np.zeros(out_c, dtype=np.float32)
        self.k, self.stride, self.pad = k, stride, pad
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x_shape = x.shape
        n, c, h, w = x.shape
        cols, oh, ow = im2col(x, self.k, self.k, self.stride, self.pad)
        self.cols = cols
        Wr = self.W.reshape(self.W.shape[0], -1)
        out = cols @ Wr.T + self.b
        out = out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)
        self.out_shape = (oh, ow)
        return out

    def backward(self, dout, lr, t):
        n, out_c, oh, ow = dout.shape
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, out_c)
        dW = (dout_r.T @ self.cols).reshape(self.W.shape)
        db = dout_r.sum(axis=0)
        Wr = self.W.reshape(out_c, -1)
        dcols = dout_r @ Wr

        n0, c, h, w = self.x_shape
        dx = np.zeros((n0, c, h + 2 * self.pad, w + 2 * self.pad), dtype=np.float32)
        dcols = dcols.reshape(n0, oh, ow, c, self.k, self.k).transpose(0, 3, 4, 5, 1, 2)
        for i in range(self.k):
            i_max = i + self.stride * oh
            for j in range(self.k):
                j_max = j + self.stride * ow
                dx[:, :, i:i_max:self.stride, j:j_max:self.stride] += dcols[:, :, i, j, :, :]
        if self.pad:
            dx = dx[:, :, self.pad:-self.pad, self.pad:-self.pad]

        self._adam(self.W, dW, self.mW, self.vW, lr, t)
        self._adam(self.b, db, self.mb, self.vb, lr, t)
        return dx

    @staticmethod
    def _adam(param, grad, m, v, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        m *= b1; m += (1 - b1) * grad
        v *= b2; v += (1 - b2) * (grad ** 2)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        param -= lr * mhat / (np.sqrt(vhat) + eps)


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout, lr, t):
        return dout * self.mask


class MaxPool2x2:
    def forward(self, x):
        n, c, h, w = x.shape
        x_r = x.reshape(n, c, h // 2, 2, w // 2, 2)
        self.x_shape = x.shape
        out = x_r.max(axis=(3, 5))
        self.mask = (x_r == out[:, :, :, None, :, None])
        return out

    def backward(self, dout, lr, t):
        n, c, h, w = self.x_shape
        dout_r = dout[:, :, :, None, :, None]
        dx = (self.mask * dout_r).reshape(n, c, h, w)
        return dx


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, lr, t):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, in_f, out_f):
        scale = np.sqrt(2.0 / in_f)
        self.W = rng.normal(0, scale, (in_f, out_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr, t):
        dW = self.x.T @ dout
        db = dout.sum(axis=0)
        dx = dout @ self.W.T
        Conv2D._adam(self.W, dW, self.mW, self.vW, lr, t)
        Conv2D._adam(self.b, db, self.mb, self.vb, lr, t)
        return dx


def softmax_ce_loss(logits, y):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-9).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    return loss, probs, dlogits


class TinyCNN:
    """Conv(1->8) -> ReLU -> Pool -> Conv(8->16) -> ReLU -> Pool -> FC(64) -> ReLU -> FC(2)"""
    def __init__(self):
        self.c1 = Conv2D(1, 8, k=3, stride=1, pad=1)
        self.r1 = ReLU()
        self.p1 = MaxPool2x2()
        self.c2 = Conv2D(8, 16, k=3, stride=1, pad=1)
        self.r2 = ReLU()
        self.p2 = MaxPool2x2()
        self.flat = Flatten()
        flat_dim = 16 * (IMG // 4) * (IMG // 4)
        self.fc1 = Dense(flat_dim, 64)
        self.r3 = ReLU()
        self.fc2 = Dense(64, 2)
        self.layers = [self.c1, self.r1, self.p1, self.c2, self.r2, self.p2,
                        self.flat, self.fc1, self.r3, self.fc2]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dlogits, lr, t):
        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d, lr, t)


def augment(img):
    """Light augmentation: random flip + small rotation, simulating the
    orientation/appearance variation the DL approach must generalise to."""
    if rng.random() < 0.5:
        img = np.fliplr(img)
    ang = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG / 2, IMG / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (IMG, IMG), borderValue=0)
    return img


def load_data():
    with open(os.path.join(DATA_DIR, "labels.csv")) as f:
        rows = list(csv.DictReader(f))
    X, y, fnames = [], [], []
    for r in rows:
        img = cv2.imread(os.path.join(IMAGES_DIR, r["filename"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG, IMG)).astype(np.float32) / 255.0
        X.append(img)
        y.append(1 if r["label"] == "DEFECT" else 0)
        fnames.append(r["filename"])
    return np.array(X), np.array(y), fnames


def run(epochs=45, batch_size=16, lr=2e-3):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix)

    X, y, fnames = load_data()
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X, y, fnames, test_size=0.30, random_state=42, stratify=y)

    net = TinyCNN()
    n = len(X_train)
    t_step = 0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    Xte = X_test[:, None, :, :]

    for epoch in range(1, epochs + 1):
        cur_lr = lr if epoch <= 30 else lr * 0.3  # simple step decay for late-stage stability
        idx = rng.permutation(n)
        losses, correct = [], 0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            imgs = np.stack([augment(X_train[i].copy()) for i in batch_idx])[:, None, :, :]
            labels = y_train[batch_idx]
            logits = net.forward(imgs)
            loss, probs, dlogits = softmax_ce_loss(logits, labels)
            t_step += 1
            net.backward(dlogits, cur_lr, t_step)
            losses.append(loss)
            correct += (probs.argmax(1) == labels).sum()
        train_loss = float(np.mean(losses))
        train_acc = correct / n

        val_logits = net.forward(Xte)
        val_loss, val_probs, _ = softmax_ce_loss(val_logits, y_test)
        val_acc = (val_probs.argmax(1) == y_test).mean()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    t0 = time.time()
    test_logits = net.forward(Xte)
    infer_ms_per_img = (time.time() - t0) / len(Xte) * 1000
    y_pred = test_logits.argmax(1)

    metrics = {
        "approach": "Deep Learning (from-scratch CNN, trained end-to-end)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "inference_ms_per_image": infer_ms_per_img,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "epochs": epochs,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dl_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(OUT_DIR, "dl_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(OUT_DIR, "dl_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true", "pred"])
        for fn, t_, p in zip(f_test, y_test, y_pred):
            w.writerow([fn, int(t_), int(p)])

    print(json.dumps(metrics, indent=2))
    return metrics, history


if __name__ == "__main__":
    run()
