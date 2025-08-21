from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io, os, json
import numpy as np

"""
This service now supports TWO backends:

A) JSON linear model (model.json) — expects 9-dim features per image
B) ONNX model (model.onnx)       — classic CNN export (optional)

Load order:
- If MODEL_LOCAL_PATH (default /app/model.onnx) exists => use ONNX
- Else, if MODEL_JSON_PATH (default /app/model.json) exists => use JSON
- Else -> error
"""

app = FastAPI(title="Disease Predictor (JSON/ONNX, top-k)", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

MODEL_ONNX_PATH = os.getenv("MODEL_LOCAL_PATH", "/app/model.onnx")
MODEL_JSON_PATH = os.getenv("MODEL_JSON_PATH", "/app/model.json")
LABELS_PATH = os.getenv("LABELS_PATH", "/app/labels.json")
TOPK = int(os.getenv("TOPK", "3"))
IMG_SZ = 224  # eval/infer resizing

# runtime state
_backend = None          # "onnx" or "json"
_labels: list[str] | None = None

# ONNX bits (lazy import)
_ort_session = None

# JSON linear bits
_W = None   # (C, 9)
_b = None   # (C,)

# ----------------- Small helpers -----------------
def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def _to_grayscale(arr_rgb: np.ndarray) -> np.ndarray:
    # arr_rgb: (H, W, 3) in [0..1]
    return (0.2989 * arr_rgb[...,0] + 0.5870 * arr_rgb[...,1] + 0.1140 * arr_rgb[...,2]).astype("float32")

def _conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # simple valid conv with zero-pad to keep same size
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(img, dtype="float32")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return out

def _extract_9_features(img_pil: Image.Image) -> np.ndarray:
    """
    9D features:
      0-2: mean R,G,B
      3-5: std  R,G,B
      6  : mean grayscale
      7  : mean gradient magnitude (Sobel)
      8  : variance grayscale
    All computed on a resized 224x224 RGB image in [0..1].
    """
    im = img_pil.convert("RGB").resize((IMG_SZ, IMG_SZ))
    arr = np.asarray(im).astype("float32") / 255.0  # (H,W,3)
    means = arr.mean(axis=(0,1))
    stds  = arr.std(axis=(0,1))

    gray = _to_grayscale(arr)
    # Sobel
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

def _preprocess_for_onnx(img_pil: Image.Image) -> np.ndarray:
    # Standard MobileNet-like preprocessing
    im = img_pil.convert("RGB").resize((IMG_SZ, IMG_SZ))
    arr = np.asarray(im).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2,0,1))[np.newaxis, ...]  # NCHW
    return arr

# ----------------- Model loading -----------------
def _load_labels():
    global _labels
    if _labels is None:
        if not os.path.exists(LABELS_PATH):
            raise RuntimeError(f"Missing labels: {LABELS_PATH}")
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            j = json.load(f)
        _labels = j["classes"]
        if not isinstance(_labels, list) or not _labels:
            raise RuntimeError("labels.json must have a non-empty 'classes' list")

def _try_load_onnx() -> bool:
    global _ort_session
    if os.path.exists(MODEL_ONNX_PATH):
        import onnxruntime as ort
        _ort_session = ort.InferenceSession(MODEL_ONNX_PATH, providers=["CPUExecutionProvider"])
        return True
    return False

def _try_load_json() -> bool:
    global _W, _b
    if os.path.exists(MODEL_JSON_PATH):
        with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
            j = json.load(f)
        W = np.array(j["W"], dtype="float32")  # (C, 9)
        b = np.array(j["b"], dtype="float32")  # (C,)
        if W.ndim != 2 or b.ndim != 1:
            raise RuntimeError("model.json must contain 2D 'W' and 1D 'b'")
        if W.shape[1] != 9:
            raise RuntimeError(f"Expected W to have 9 columns (got {W.shape[1]})")
        _W, _b = W, b
        return True
    return False

def _ensure_loaded():
    global _backend
    if _backend is not None:
        return
    _load_labels()
    if _try_load_onnx():
        _backend = "onnx"
        return
    if _try_load_json():
        _backend = "json"
        return
    raise RuntimeError(f"No model found. Looked for {MODEL_ONNX_PATH} (ONNX) and {MODEL_JSON_PATH} (JSON).")

# ----------------- Schemas -----------------
class TopPred(BaseModel):
    label: str
    confidence: float

class Prediction(BaseModel):
    top1: TopPred
    topk: list[TopPred]
    model_version: str | None = None
    backend: str

# ----------------- Routes -----------------
@app.get("/health")
def health():
    which = "none"
    try:
        _ensure_loaded()
        which = _backend
    except Exception:
        pass
    return {"ok": True, "backend": which}

@app.get("/labels")
def labels():
    _ensure_loaded()
    return {"classes": _labels}

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg","image/png"):
        raise HTTPException(400, "Only JPEG/PNG images supported")
    _ensure_loaded()

    img = Image.open(io.BytesIO(await file.read()))
    if _backend == "onnx":
        x = _preprocess_for_onnx(img)
        inp_name = _ort_session.get_inputs()[0].name
        logits = _ort_session.run(None, {inp_name: x})[0][0]  # (C,)
    else:
        x9 = _extract_9_features(img)  # (9,)
        logits = (_W @ x9) + _b       # (C,)

    probs = _softmax_1d(logits)
    idxs = np.argsort(-probs)[:TOPK]
    topk = [TopPred(label=_labels[i], confidence=float(probs[i])) for i in idxs]
    return Prediction(
        top1=topk[0],
        topk=topk,
        model_version="linear-9f-json" if _backend=="json" else "onnx",
        backend=_backend
    )
