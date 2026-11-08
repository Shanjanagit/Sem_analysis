"""
SEM Image Prediction from .arrow files (simple version)
Models: CNN, LSTM, Transformer + cosine similarity to class prototypes.

Reads Hugging Face-style .arrow dataset files (images stored as bytes).
Requires: pip install pyarrow

Just edit the SETTINGS block below and run:
    python sem_predict_simple.py
"""

import io
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
from PIL import Image as PILImage
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from model_build.logs import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "database"
logger.info(DB_DIR)

if str(DB_DIR) not in sys.path:
    sys.path.insert(0, str(DB_DIR))

from db_insertion import insert_prediction

RESULTS_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "sem_prediction.log"

logger = logging.getLogger("SEM_Prediction")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)




# ============================================================
# SETTINGS  (edit these)
# ============================================================

INPUT_DIR  = Path(r"C:\Users\VIGNESH\Desktop\SEM_analysis\test")   # folder with the .arrow files
RESULTS_DIR = Path(r"C:\Users\VIGNESH\Desktop\SEM_analysis\results_1\updated")   # models + prototypes live here
OUTPUT_CSV = RESULTS_DIR / "predictions_1.csv"

MODELS = ["CNN", "LSTM", "Transformer"]   # remove any you don't want to run
# MODELS = ["CNN"]

MIN_CONFIDENCE = 0.70
MIN_MARGIN = 0.50
TOP_K = 3

IMG_SIZE = 224
PATCH_SIZE = 7
EMBED_DIM = 128
COSINE_LOGIT_SCALE = 10.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODELS (must match training)
# ============================================================

class CosineClassifier(nn.Module):
    def __init__(self, embedding_dim, n_classes, scale=COSINE_LOGIT_SCALE):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x, w)


class SEMCNN(nn.Module):
    def __init__(self, num_classes, embedding_dim=EMBED_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(), nn.Linear(192, embedding_dim), nn.GELU(), nn.Dropout(0.2)
        )
        self.cosine_classifier = CosineClassifier(embedding_dim, num_classes)

    def forward_features(self, x):
        return self.embedding(self.features(x))

    def forward(self, x):
        return self.cosine_classifier(self.forward_features(x))


class SEMLSTM(nn.Module):
    def __init__(self, num_classes, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                 hidden_size=128, embedding_dim=EMBED_DIM):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("IMG_SIZE must be divisible by PATCH_SIZE.")
        self.patch_size = patch_size
        self.lstm = nn.LSTM(patch_size * patch_size, hidden_size,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim), nn.GELU(), nn.Dropout(0.2)
        )
        self.cosine_classifier = CosineClassifier(embedding_dim, num_classes)

    def forward_features(self, x):
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(x.size(0), -1, p * p)
        seq, _ = self.lstm(patches)
        return self.embedding(seq[:, -1, :])

    def forward(self, x):
        return self.cosine_classifier(self.forward_features(x))


class SEMTransformer(nn.Module):
    def __init__(self, num_classes, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                 embed_dim=EMBED_DIM, num_heads=4, num_layers=3):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("IMG_SIZE must be divisible by PATCH_SIZE.")
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.cosine_classifier = CosineClassifier(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = self.encoder(torch.cat([cls, x], dim=1) + self.pos_embed)
        return self.norm(x[:, 0])

    def forward(self, x):
        return self.cosine_classifier(self.forward_features(x))


# ============================================================
# HELPERS
# ============================================================

def _batches(path):
    """Yield record batches from a .arrow file (stream or file format)."""
    mm = pa.memory_map(str(path), "r")
    try:
        reader = pa.ipc.open_stream(mm)
        for batch in reader:
            yield batch
    except pa.ArrowInvalid:
        mm = pa.memory_map(str(path), "r")
        reader = pa.ipc.open_file(mm)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def _label_names(input_dir, schema):
    """Get label names (index -> name) from dataset_info.json or arrow metadata."""
    feats = None
    info_path = Path(input_dir) / "dataset_info.json"
    try:
        if info_path.is_file():
            feats = json.loads(info_path.read_text(encoding="utf-8")).get("features")
        elif schema.metadata and b"huggingface" in schema.metadata:
            meta = json.loads(schema.metadata[b"huggingface"].decode("utf-8"))
            feats = meta.get("info", {}).get("features")
    except Exception:
        feats = None
    if feats:
        for col, f in feats.items():
            if isinstance(f, dict) and "names" in f:
                return col, list(f["names"])
    return None, None


def _find_columns(schema):
    image_col = label_col = None
    for field in schema:
        t = field.type
        if image_col is None and (
            (pa.types.is_struct(t) and "bytes" in [t.field(i).name for i in range(t.num_fields)])
            or pa.types.is_binary(t) or pa.types.is_large_binary(t)
        ):
            image_col = field.name
        if label_col is None and field.name.lower() in ("label", "labels", "class", "target") \
                and (pa.types.is_integer(t) or pa.types.is_string(t)):
            label_col = field.name
    return image_col, label_col


def _decode(value):
    """Arrow image cell -> grayscale uint8 numpy array."""
    if isinstance(value, dict):
        data, path = value.get("bytes"), value.get("path")
        src = io.BytesIO(data) if data is not None else path
    else:
        src = io.BytesIO(value)
    with PILImage.open(src) as im:
        return np.asarray(im.convert("L"))


def iter_arrow_samples(input_dir):
    """Yield (sample_id, gray_image_array, true_label_or_None)."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.rglob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No .arrow files found in {input_dir}")

    for f in files:
        row_no = 0
        image_col = label_col = names = None
        for batch in _batches(f):
            if image_col is None:
                image_col, label_col = _find_columns(batch.schema)
                if image_col is None:
                    raise ValueError(f"No image column found in {f.name}. "
                                     f"Columns: {batch.schema.names}")
                _, names = _label_names(f.parent, batch.schema)
            images = batch.column(image_col).to_pylist()
            labels = batch.column(label_col).to_pylist() if label_col else [None] * len(images)
            for img, lab in zip(images, labels):
                if lab is not None and names is not None and isinstance(lab, int):
                    lab = names[lab]
                yield f"{f.name}#{row_no}", _decode(img), (None if lab is None else str(lab))
                row_no += 1


def preprocess_image(gray, img_size):
    """Grayscale numpy array -> model input tensor [1,1,H,W]."""
    img = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0)


def load_model(name):
    """Load model + prototypes ONCE (not per image)."""
    ckpt_path = RESULTS_DIR / f"{name.lower()}_cosine_best.pt"
    proto_path = RESULTS_DIR / f"{name.lower()}_class_prototypes.npy"

    for p in (ckpt_path, proto_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing file: {p}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", IMG_SIZE))
    patch_size = int(ckpt.get("patch_size", PATCH_SIZE))
    emb_dim = int(ckpt.get("embedding_dim", EMBED_DIM))
    n = len(class_names)

    if name == "CNN":
        model = SEMCNN(n, embedding_dim=emb_dim)
    elif name == "LSTM":
        model = SEMLSTM(n, img_size=img_size, patch_size=patch_size, embedding_dim=emb_dim)
    elif name == "Transformer":
        model = SEMTransformer(n, img_size=img_size, patch_size=patch_size, embed_dim=emb_dim)
    else:
        raise ValueError(f"Unsupported model: {name}")

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE).eval()

    protos = np.load(proto_path).astype(np.float32)
    if protos.shape != (n, emb_dim):
        raise ValueError(f"Prototype shape {protos.shape}, expected {(n, emb_dim)}")
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12

    return model, protos, class_names, img_size


@torch.inference_mode()
def predict(model, protos, img_size, gray):
    x = preprocess_image(gray, img_size).to(DEVICE)
    emb = F.normalize(model.forward_features(x), p=2, dim=1).cpu().numpy()[0]
    sims = emb @ protos.T
    order = np.argsort(sims)[::-1]
    return sims, order


# ============================================================
# MAIN
# ============================================================

def main(input_dir=None, output_csv=None):
    input_dir = Path(input_dir) if input_dir else INPUT_DIR
    output_csv = Path(output_csv) if output_csv else OUTPUT_CSV
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Input : {INPUT_DIR}")
    logger.info(f"Output: {OUTPUT_CSV}\n")

    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(f"Input folder not found: {INPUT_DIR}")

    # Load every model once
    loaded = {}
    for name in MODELS:
        try:
            loaded[name] = load_model(name)
            logger.info(f"Loaded {name}")
        except Exception as e:
            logger.info(f"Skipping {name}: {e}")
    if not loaded:
        raise RuntimeError("No models could be loaded.")
    

    rows = []
    n_samples = 0

    # Read the arrow files once; run every model on each sample
    for sample_id, gray, true_label in iter_arrow_samples(INPUT_DIR):
        n_samples += 1
        for name, (model, protos, class_names, img_size) in loaded.items():
            row = {"sample": sample_id, "true_label": true_label, "model": name}
            try:
                sims, order = predict(model, protos, img_size, gray)
                best = int(order[0])
                best_score = float(sims[order[0]])
                second_score = float(sims[order[1]])
                created_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                row["predicted_class"] = class_names[best]
                row["cosine_similarity"] = float(sims[best])

                margin = best_score - second_score
                if (
                    best_score >= MIN_CONFIDENCE and
                    margin >= MIN_MARGIN
                ):
                    logger.info("Insertion the predictions in the db..")
                    insert_prediction(
                        class_names[order[0]],
                        best_score,
                        created_date
                    )
                    logger.info("Prediction results inserted in the db")
                    row["accepted"] = True
                else:
                    row["accepted"] = False

                if true_label is not None:
                    row["correct"] = class_names[best] == true_label

                for r, idx in enumerate(order[:TOP_K], start=1):
                    row[f"top{r}_class"] = class_names[int(idx)]
                    row[f"top{r}_similarity"] = float(sims[int(idx)])
                for i, cname in enumerate(class_names):
                    row[f"sim_{cname}"] = float(sims[i])
            except Exception as e:
                row["predicted_class"] = "ERROR"
                row["error"] = str(e)
            rows.append(row)

        if n_samples % 50 == 0:
            logger.info(f"Processed {n_samples} samples...")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info(f"\nDone: {n_samples} samples x {len(loaded)} model(s)")
    if "correct" in df.columns:
        logger.info("\nAccuracy per model (only meaningful if label names match training):")
        logger.info(df.groupby("model")["correct"].mean().round(4).to_string())
    logger.info(f"\nPredictions saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main(input_dir=None, output_csv=None)