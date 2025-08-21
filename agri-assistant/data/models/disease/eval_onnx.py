import argparse, os, json, glob, io
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt

# ------------- image preprocessing (must match training/export) -----------------
IMG_SZ = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD  = np.array([0.229, 0.224, 0.225], dtype="float32")

def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SZ, IMG_SZ))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2,0,1))[np.newaxis, ...]  # NCHW
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def load_labels(labels_json_path: str) -> list[str]:
    with open(labels_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "classes" in data:
        return list(data["classes"])
    if isinstance(data, list):
        return data
    raise ValueError("labels.json must be either {\"classes\":[...]} or a JSON array of strings")

def iter_val_images(val_root: str, class_to_idx: dict[str,int]):
    """
    Yields (filepath, class_index) for every image under val_root\<class>\*.jpg|png|jpeg
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    for cls, idx in class_to_idx.items():
        cls_dir = Path(val_root) / cls
        for pat in exts:
            for p in cls_dir.glob(pat):
                yield str(p), idx

def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: str):
    fig = plt.figure(figsize=(max(8, len(class_names)*0.6), max(6, len(class_names)*0.6)))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True, help=r"path to datasets\plantvillage\val")
    ap.add_argument("--model", default=r"services\disease-predictor\app\model.onnx")
    ap.add_argument("--labels", default=r"services\disease-predictor\app\labels.json")
    ap.add_argument("--outdir", default=r"data\models\disease\eval_out")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # labels & class mapping
    classes = load_labels(args.labels)
    class_to_idx = {c:i for i,c in enumerate(classes)}

    # ONNX runtime session
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # collect data
    xs = []
    y_true = []
    filepaths = []
    for fp, yi in iter_val_images(args.val_dir, class_to_idx):
        try:
            img = Image.open(fp)
            x = preprocess(img)
            xs.append(x)
            y_true.append(yi)
            filepaths.append(fp)
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")

    if not xs:
        raise SystemExit("No images found under --val-dir with the expected <class>/*.jpg|png pattern.")

    # batched inference
    y_true = np.array(y_true, dtype=np.int64)
    logits_all = []
    for i in tqdm(range(0, len(xs), args.batch), desc="Infer"):
        batch = np.concatenate(xs[i:i+args.batch], axis=0)  # (N,3,H,W)
        logits = sess.run(None, {input_name: batch})[0]     # (N,C)
        logits_all.append(logits)
    logits = np.concatenate(logits_all, axis=0)
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)

    # metrics
    top1_acc = accuracy_score(y_true, y_pred)
    # top-k
    topk_acc = top_k_accuracy_score(y_true, probs, k=min(args.topk, probs.shape[1]))
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # per-class accuracy
    per_class = {}
    for i, cls in enumerate(classes):
        idxs = np.where(y_true == i)[0]
        correct = (y_pred[idxs] == i).sum()
        total = len(idxs)
        per_class[cls] = float(correct) / total if total > 0 else 0.0

    # save outputs
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "top1_accuracy": top1_acc,
            "top{}_accuracy".format(args.topk): topk_acc,
            "per_class_accuracy": per_class,
            "num_samples": int(len(y_true))
        }, f, indent=2)

    with open(os.path.join(args.outdir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion(cm, classes, os.path.join(args.outdir, "confusion_matrix.png"))

    # quick console print
    print("\n==== EVAL SUMMARY ====")
    print(f"samples         : {len(y_true)}")
    print(f"top-1 accuracy  : {top1_acc:.4f}")
    print(f"top-{min(args.topk, probs.shape[1])} accuracy: {topk_acc:.4f}")
    print("\nPer-class accuracy:")
    for k,v in per_class.items():
        print(f"  {k:20s} {v:.4f}")
    print("\nClassification report saved to:", os.path.join(args.outdir, "classification_report.txt"))
    print("Confusion matrix saved to    :", os.path.join(args.outdir, "confusion_matrix.png"))
    print("Metrics JSON saved to        :", os.path.join(args.outdir, "metrics.json"))

if __name__ == "__main__":
    main()
