"""
Train a crop-disease classifier with transfer learning (MobileNetV3 Small),
export ONNX + labels.json so the FastAPI service can load them.

DATA LAYOUT (choose one):
  direct class folders, like datasets/raw/:
    healthy/
    powdery_mildew/
    rust/
  or split folders:
    datasets/
      train/<class>/
      val/<class>/  # optional; if absent, a deterministic split is made from train/

USAGE (from `agri-assistant/`):
  python -m venv .venv
  Activate with .\.venv\Scripts\Activate.ps1 on PowerShell,
  or source .venv/bin/activate on macOS/Linux.
  python -m pip install -r data/models/disease/requirements.txt
  python data/models/disease/train.py --data-dir datasets/raw --epochs 10

Artifacts will be written to:
  services/disease-predictor/app/model.onnx
  services/disease-predictor/app/labels.json
"""
import argparse, json, os, random, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def build_loaders(data_dir: Path, img_sz=224, batch=64, val_split=0.15):
    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_sz, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_sz*1.15)),
        transforms.CenterCrop(img_sz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    split_train_root = data_dir/"train"
    val_root = data_dir/"val"
    has_train_split = split_train_root.is_dir()

    if has_train_split:
        train_root = split_train_root
        has_val = val_root.is_dir()
    else:
        if val_root.exists():
            raise ValueError(f"Expected a `train/` directory beside `{val_root}` when `val/` exists")
        train_root = data_dir
        has_val = False

    if has_val:
        train_ds = datasets.ImageFolder(train_root, transform=train_tf)
        val_ds = datasets.ImageFolder(val_root, transform=eval_tf)
        if train_ds.classes != val_ds.classes:
            raise ValueError("train/ and val/ must contain the same class directories")
        classes = train_ds.classes
    else:
        full = datasets.ImageFolder(train_root, transform=train_tf)
        if len(full) < 2:
            raise ValueError("At least two labeled images are required to create a train/validation split")
        n_val = min(len(full)-1, max(1, int(len(full)*val_split)))
        n_train = len(full)-n_val
        train_ds, val_split_ds = random_split(full, [n_train, n_val],
                                               generator=torch.Generator().manual_seed(42))
        eval_full = datasets.ImageFolder(train_root, transform=eval_tf)
        val_ds = Subset(eval_full, val_split_ds.indices)
        classes = full.classes

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, classes

def build_model(num_classes):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x,y in loader:
        x = x.to(device)
        logits = model(x).cpu()
        all_logits.append(logits); all_y.append(y)
    logits = torch.cat(all_logits); y = torch.cat(all_y)
    prob = logits.softmax(1)
    acc = (prob.argmax(1)==y).float().mean().item()
    return acc, y.numpy(), prob.numpy()

def train(args):
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, classes = build_loaders(Path(args.data_dir), img_sz=args.img_sz, batch=args.batch)
    model = build_model(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss = nn.CrossEntropyLoss()

    best_acc, best_w = -1.0, None
    for epoch in range(1, args.epochs+1):
        model.train(); tot=0
        pbar = tqdm(tr, desc=f"epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            l = loss(logits, y); l.backward(); opt.step()
            tot += l.item(); pbar.set_postfix(loss=f"{tot/len(tr):.4f}")
        sched.step()

        va_acc, y_true, y_prob = evaluate(model, va, device)
        if va_acc > best_acc:
            best_acc = va_acc
            best_w = {k:v.cpu() for k,v in model.state_dict().items()}
        print(f"[val] acc={va_acc:.4f} best={best_acc:.4f}")

    # load best and final eval
    model.load_state_dict(best_w)
    va_acc, y_true, y_prob = evaluate(model, va, device)
    y_pred = y_prob.argmax(1)
    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # export ONNX + labels.json into service folder
    service_dir = Path("services/disease-predictor/app")
    service_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1,3,args.img_sz,args.img_sz, device=device)
    onnx_path = service_dir/"model.onnx"
    torch.onnx.export(model, dummy, onnx_path.as_posix(),
                      input_names=["input"], output_names=["logits"],
                      opset_version=13, do_constant_folding=True)
    with open(service_dir/"labels.json", "w") as f:
        json.dump({"classes": classes, "img_sz": args.img_sz}, f)
    print(f"Exported: {onnx_path} and labels.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="class folders directly or train/ (and optional val/)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-sz", type=int, default=224)
    args = ap.parse_args()
    train(args)
