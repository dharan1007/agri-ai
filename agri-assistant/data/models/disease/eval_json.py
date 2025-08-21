# data/models/disease/eval_json.py

import argparse, json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# --------- feature extractor (must match the API logic) ----------
IMG_SZ = 224

def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.zeros_like(x)

def _to_grayscale(arr_rgb: np.ndarray) -> np.ndarray:
    return (0.2989 * arr_rgb[...,0] + 0.5870 * arr_rgb[...,1] + 0.1140 * arr_rgb[...,2]).astype("float32")

def _conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(img, dtype="float32")
    H, W = img.shape
    for i in range(H):
        row = i + ph
        # small perf tweak: slice once per row
        row_slice = padded[row - ph: row + ph + 1]
        for j in range(W):
            col = j + pw
            region = row_slice[:, col - pw: col + pw + 1]
            out[i, j] = float(np.sum(region * kernel))
    return out

def extract_9_features(img_pil: Image.Image) -> np.ndarray:
    im = img_pil.convert("RGB").resize((IMG_SZ, IMG_SZ))
    arr = np.asarray(im).astype("float32") / 255.0  # (H,W,3)
    means = arr.mean(axis=(0,1))
    stds  = arr.std(axis=(0,1))
    gray = _to_grayscale(arr)
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype="float32")
    ky = np.array([[ 1,2,1],[ 0,0,0],[-1,-2,-1]], dtype="float32")
    gx = _conv2d_same(gray, kx)
    gy = _conv2d_same(gray, ky)
    grad = np.sqrt(gx*gx + gy*gy)
    mean_grad = float(grad.mean())

    feat = np.zeros((9,), dtype="float32")
    feat[0:3] = means
    feat[3:6] = stds
    feat[6]   = float(gray.mean())
    feat[7]   = mean_grad
    feat[8]   = float(gray.var())
    return feat

# --------- evaluation helpers ----------
VALID_EXTS = {".jpg", ".jpeg", ".png"}

def iter_images(root: Path):
    """
    Expects: datasets/raw/<class>/*.(jpg|jpeg|png)
    """
    for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        label = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                yield p, label

def main():
    ap = argparse.ArgumentParser(description="Evaluate simple JSON linear model on a folder-per-class dataset.")
    ap.add_argument("--val-dir", required=False, default="datasets/raw",
                    help="Root folder with class subfolders (default: datasets/raw)")
    ap.add_argument("--model-json", required=True, help="Path to model.json (with keys 'W' and 'b').")
    ap.add_argument("--labels", required=True, help="labels.json with field {'classes': [...]}")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    # load model
    with open(args.model_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    W = np.array(m["W"], dtype="float32")  # (C, 9)
    B = np.array(m["b"], dtype="float32")  # (C,)

    # load labels (order matters: must align with rows of W and elements of b)
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)["classes"]
    C = len(labels)
    assert W.shape[0] == C, f"W rows ({W.shape[0]}) must equal #classes ({C})"
    assert W.shape[1] == 9, f"W must have 9 columns (got {W.shape[1]})"
    assert B.shape[0] == C, "b must have C elements"

    label_set = set(labels)
    val_root = Path(args.val_dir)

    total = 0
    top1_correct = 0
    topk_correct = 0
    skipped_unknown = 0

    per_class_tot  = {c: 0 for c in labels}
    per_class_top1 = {c: 0 for c in labels}

    for path, true_label in tqdm(iter_images(val_root), desc="Eval"):
        # skip folders whose name isn't in labels.json
        if true_label not in label_set:
            skipped_unknown += 1
            continue

        try:
            img = Image.open(path)
        except Exception:
            continue

        x9 = extract_9_features(img)         # (9,)
        logits = (W @ x9) + B                # (C,)
        probs = _softmax_1d(logits)
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        top_idx = np.argsort(-probs)[:args.topk]
        top_labels = [labels[i] for i in top_idx]

        total += 1
        per_class_tot[true_label] += 1

        if pred_label == true_label:
            top1_correct += 1
            per_class_top1[true_label] += 1
        if true_label in top_labels:
            topk_correct += 1

    top1 = (top1_correct / total) if total else 0.0
    topk = (topk_correct / total) if total else 0.0

    print("\n--- RESULTS ---")
    print(f"Images evaluated: {total}")
    print(f"Skipped (unknown class not in labels.json): {skipped_unknown}")
    print(f"Top-1 accuracy:  {top1:.4f}")
    print(f"Top-{args.topk} acc:   {topk:.4f}")

    print("\nPer-class Top-1:")
    for cls in labels:
        n = per_class_tot.get(cls, 0)
        c = per_class_top1.get(cls, 0)
        acc = (c / n) if n else 0.0
        print(f"  {cls:20s}  n={n:5d}  acc={acc:.4f}")

if __name__ == "__main__":
    main()
