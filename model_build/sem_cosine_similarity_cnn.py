# SEM image classification with CNN, LSTM, Transformer + cosine-similarity prototypes

# Google Colab: run this AFTER dataset loading and dataset analysis are complete.

# Expected existing variables:

#   dataset      : Hugging Face DatasetDict, containing split "train"

#   image_col    : usually "image"

#   label_col    : usually "label"

#   DEVICE       : torch.device("cuda" if torch.cuda.is_available() else "cpu")

#   OUTPUT_DIR   : output folder in Google Drive (or set below)

#

# This pipeline:

# 1) makes stratified train/validation/test splits

# 2) trains CNN, LSTM, and Transformer feature extractors with a cosine-similarity classifier

# 3) extracts train embeddings and builds one class prototype per category

# 4) predicts validation/test labels using cosine similarity to class prototypes

# 5) saves checkpoints, reports, confusion matrices, embeddings, and comparison results



import os

import json

import time

import random

import io

from collections import Counter



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

from PIL import Image as PILImage

from datasets import Dataset as HFDataset, DatasetDict, concatenate_datasets, Image as HFImage



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import logging

from torch.utils.data import Dataset as TorchDataset, DataLoader




from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (

    accuracy_score, precision_score, recall_score, f1_score,

    classification_report, confusion_matrix

)



try:
    from logs import logger
except ImportError:
    logger = logging.getLogger("sem_pipeline_cnn_lstm_transformers")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")



# -------------------------

# A. Configuration

# -------------------------

IMG_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 20


LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 12

SCHEDULER_PATIENCE = 4

NUM_WORKERS = 0

PATCH_SIZE = 7

EMBED_DIM = 128

COSINE_LOGIT_SCALE = 10.0



# Load the dataset directly from its existing path

import os
import glob

DATASET_PATH = r"C:\Users\VIGNESH\Desktop\SEM_analysis\train"

OUTPUT_DIR = r"C:\Users\VIGNESH\Desktop\SEM_analysis\results_1\updated"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Find all locally available Arrow shards

arrow_files = sorted(

    glob.glob(os.path.join(DATASET_PATH, "data-*.arrow"))

)



if not arrow_files:

    raise FileNotFoundError(

        f"No Arrow shards found in: {DATASET_PATH}"

    )



logger.info(f"Found {len(arrow_files)} Arrow shards.")
if len(arrow_files) < 34:
    logger.warning(f"Only {len(arrow_files)} Arrow shards found; the full dataset was expected to contain 34. Training will use only the shards present in this folder.")



# Load each shard directly

shards = []



for file_path in arrow_files:

    logger.info(f"Loading: {os.path.basename(file_path)}")

    shard = HFDataset.from_file(file_path)

    shards.append(shard)

    logger.info(f"Rows: {len(shard)}")



# Combine the available shards

if len(shards) == 1:

    train_dataset = shards[0]

else:

    train_dataset = concatenate_datasets(shards)



logger.info("Dataset loaded successfully!")

logger.info(f"Available samples: {len(train_dataset)}")

logger.info(f"Columns: {train_dataset.column_names}")





# Try loading the saved Hugging Face dataset

try:

    # Keep this variable if the rest of your pipeline expects `dataset`

    dataset = train_dataset

    logger.info(f"Dataset loaded from: {DATASET_PATH}")



except (FileNotFoundError, ValueError) as e:

    raise RuntimeError(

        "The folder contains Arrow shards, but it is not "

        "a valid save_to_disk dataset directory. The dataset "

        "metadata or other shards may be missing. "

        "Check the original dataset folder."

    ) from e



# Get training data

if isinstance(dataset, DatasetDict):

    train_dataset = dataset["train"]

else:

    train_dataset = dataset



logger.info(f"Training samples: {len(train_dataset)}")

logger.info(f"Columns: {train_dataset.column_names}")





DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Device:{DEVICE}")

if DEVICE.type == "cuda":

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")



# -------------------------

# B. Validate existing dataset variables and decode images

# -------------------------

if "dataset" not in globals():

    raise NameError("Load the Hugging Face dataset first and ensure it is named `dataset`.")



# first_split_name = list(dataset.keys())[0]

# first_split = dataset[first_split_name]



# # The dataset described earlier has columns: image_filename, label, image.

# image_col = globals().get("image_col", "image")

# label_col = globals().get("label_col", "label")



# Handle both DatasetDict and Dataset

if isinstance(dataset, DatasetDict):

    first_split_name = list(dataset.keys())[0]

    first_split = dataset[first_split_name]

    logger.info(f"Selected split: {first_split_name}")



elif isinstance(dataset, HFDataset):

    first_split_name = "train"

    first_split = dataset

    logger.info("Using the loaded dataset directly as the training split.")



else:

    raise TypeError(

        f"Unexpected dataset type: {type(dataset).__name__}"

    )



logger.info(f"Number of samples: {len(first_split)}")

logger.info(f"Columns: {first_split.column_names}")



# Dataset columns

image_col = "image"

label_col = "label"



# Verify that the columns exist

available_columns = first_split.column_names



if image_col not in available_columns:

    raise ValueError(f"Image column '{image_col}' not found.")



if label_col not in available_columns:

    raise ValueError(f"Label column '{label_col}' not found.")



logger.info(f"Image column:{image_col}")

logger.info(f"Label column:{label_col}")

logger.info(f"Available columns:{available_columns}")



# if image_col not in first_split.column_names:

#     image_col = next((c for c in first_split.column_names

#                       if type(first_split.features[c]).__name__.lower() == "image"), None)

# if label_col not in first_split.column_names:

#     label_col = next((c for c in ["label", "labels", "class", "category", "target"]

#                       if c in first_split.column_names), None)

# if image_col is None or label_col is None:

#     raise ValueError(f"Could not identify image/label columns. Found: {first_split.column_names}")



# # Ensure the HF Image feature decodes to PIL images (the earlier inspection showed decode=False).

# for split_name in dataset.keys():

#     if type(dataset[split_name].features[image_col]).__name__.lower() == "image":

#         dataset[split_name] = dataset[split_name].cast_column(image_col, HFImage(decode=True))



# logger.info("Image column:", image_col)

# logger.info("Label column:", label_col)

# logger.info("Rows:", len(dataset[first_split_name]))



# -------------------------

# C. Gather records and stratified split

# -------------------------

records = []
split_datasets = {"train": first_split}
for split_name, split_ds in split_datasets.items():
    label_feature = split_ds.features[label_col]
    for row_idx, value in enumerate(split_ds[label_col]):
        if hasattr(label_feature, "names") and isinstance(value, (int, np.integer)):
            value = label_feature.names[int(value)]
        if value is None:
            continue
        records.append((split_name, row_idx, str(value)))

if not records:

    raise ValueError("No labelled records found.")



class_names = sorted({r[2] for r in records})

num_classes = len(class_names)

label_to_idx = {name: i for i, name in enumerate(class_names)}

idx_to_label = {i: name for name, i in label_to_idx.items()}

encoded_labels = np.asarray([label_to_idx[r[2]] for r in records], dtype=np.int64)

all_indices = np.arange(len(records))



# 70% train, 15% validation, 15% test. No fixed random seed is set.

train_idx, temp_idx = train_test_split(

    all_indices, test_size=0.30, stratify=encoded_labels

)

val_idx, test_idx = train_test_split(

    temp_idx, test_size=0.50, stratify=encoded_labels[temp_idx]

)



logger.info(f"\nClasses:{class_names}")

logger.info(f"Number of classes:{num_classes}")

logger.info(f"Split sizes — train: {len(train_idx)}, validation: {len(val_idx)}, test: {len(test_idx)}")

for split_name, idxs in [("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]:

    counts = Counter(records[int(i)][2] for i in idxs)

    logger.info(f"\n{split_name} class counts:")

    for name in class_names:

        logger.info(f"  {name}: {counts[name]}")



# -------------------------

# D. Image preprocessing and PyTorch Dataset

# -------------------------

def image_to_grayscale(image_value):

    """Convert a decoded PIL image, image dict, path, or array to grayscale uint8."""

    if isinstance(image_value, PILImage.Image):

        img = image_value

    elif isinstance(image_value, dict):

        # Prefer embedded bytes; relative filenames may not exist as local files.

        if image_value.get("bytes"):

            img = PILImage.open(io.BytesIO(image_value["bytes"]))

        elif image_value.get("path") and os.path.isfile(image_value["path"]):

            img = PILImage.open(image_value["path"])

        else:

            raise FileNotFoundError(

                f"Cannot decode image dict. Keys={list(image_value.keys())}; "

                "ensure the Hugging Face image feature uses decode=True."

            )

    elif isinstance(image_value, str):

        if not os.path.isfile(image_value):

            raise FileNotFoundError(f"Image path does not exist: {image_value}")

        img = PILImage.open(image_value)

    else:

        arr = np.asarray(image_value)

        if arr.ndim == 3:

            if arr.shape[-1] in (3, 4):

                arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)

            elif arr.shape[0] in (3, 4):

                arr = np.moveaxis(arr[:3], 0, -1)

                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        if arr.ndim != 2:

            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.dtype != np.uint8:

            arr = np.nan_to_num(arr)

            if arr.max() <= 1.0:

                arr = arr * 255.0

            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    return np.asarray(img.convert("L"))



class HFSEMImageDataset(TorchDataset):

    def __init__(self, record_indices, augment=False):

        self.record_indices = list(map(int, record_indices))

        self.augment = augment



    def __len__(self):

        return len(self.record_indices)



    def __getitem__(self, item):

        rec_idx = self.record_indices[item]

        source_split, row_idx, label_text = records[rec_idx]

        row = split_datasets[source_split][row_idx]

        img = image_to_grayscale(row[image_col])



        # Resize to a square for the three model baselines.

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)



        if self.augment:

            # Geometric augmentation suitable only if these transformations preserve class meaning.

            if random.random() < 0.5:

                img = np.fliplr(img).copy()

            if random.random() < 0.5:

                img = np.flipud(img).copy()

            if random.random() < 0.25:

                img = np.rot90(img).copy()



        img = img.astype(np.float32) / 255.0

        # Standardize grayscale intensity to roughly zero mean / unit scale.

        img = (img - 0.5) / 0.5

        x = torch.from_numpy(img.copy()).float().unsqueeze(0)

        y = torch.tensor(label_to_idx[label_text], dtype=torch.long)

        return x, y



train_ds = HFSEMImageDataset(train_idx, augment=True)

train_eval_ds = HFSEMImageDataset(train_idx, augment=False)

val_ds = HFSEMImageDataset(val_idx, augment=False)

test_ds = HFSEMImageDataset(test_idx, augment=False)



pin_memory = DEVICE.type == "cuda"

loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin_memory)

train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **loader_kwargs)

val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)



# Quick data sanity check before training.

sample_x, sample_y = next(iter(train_loader))

logger.info("\nBatch shape:", tuple(sample_x.shape), "labels shape:", tuple(sample_y.shape))

logger.info(f"Pixel range: {float(sample_x.min())} to {float(sample_x.max())}")

assert sample_x.ndim == 4 and sample_x.shape[1] == 1



# -------------------------

# E. Cosine-similarity classifier

# -------------------------

class CosineClassifier(nn.Module):

    """Learnable class vectors; logits are scaled cosine similarities."""

    def __init__(self, embedding_dim, n_classes, scale=COSINE_LOGIT_SCALE):

        super().__init__()

        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))

        nn.init.xavier_uniform_(self.weight)

        self.scale = scale



    def forward(self, embeddings):

        embeddings = F.normalize(embeddings, p=2, dim=1)

        class_vectors = F.normalize(self.weight, p=2, dim=1)

        return self.scale * F.linear(embeddings, class_vectors)



# Each model returns an embedding vector. Its classifier compares that vector

# with normalized learned class vectors using cosine similarity.

class SEMCNN(nn.Module):

    def __init__(self, num_classes, embedding_dim=EMBED_DIM):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.GELU(),

            nn.AdaptiveAvgPool2d(1)

        )

        self.embedding = nn.Sequential(

            nn.Flatten(), nn.Linear(192, embedding_dim), nn.GELU(),

            nn.Dropout(0.2)

        )

        self.cosine_classifier = CosineClassifier(embedding_dim, num_classes)



    def forward_features(self, x):

        return self.embedding(self.features(x))



    def forward(self, x):

        z = self.forward_features(x)

        return self.cosine_classifier(z)



class SEMLSTM(nn.Module):

    def __init__(self, num_classes, img_size=IMG_SIZE, patch_size=PATCH_SIZE,

                 hidden_size=128, embedding_dim=EMBED_DIM):

        super().__init__()

        if img_size % patch_size:

            raise ValueError("IMG_SIZE must be divisible by PATCH_SIZE.")

        patch_dim = patch_size * patch_size

        self.patch_size = patch_size

        self.lstm = nn.LSTM(

            input_size=patch_dim, hidden_size=hidden_size,

            num_layers=2, batch_first=True, dropout=0.2

        )

        self.embedding = nn.Sequential(

            nn.Linear(hidden_size, embedding_dim), nn.GELU(), nn.Dropout(0.2)

        )

        self.cosine_classifier = CosineClassifier(embedding_dim, num_classes)



    def forward_features(self, x):

        # [B, 1, H, W] -> ordered sequence of non-overlapping image patches

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(

            3, self.patch_size, self.patch_size

        )

        patches = patches.contiguous().view(x.size(0), -1, self.patch_size ** 2)

        sequence, _ = self.lstm(patches)

        return self.embedding(sequence[:, -1, :])



    def forward(self, x):

        return self.cosine_classifier(self.forward_features(x))



class SEMTransformer(nn.Module):

    def __init__(self, num_classes, img_size=IMG_SIZE, patch_size=PATCH_SIZE,

                 embed_dim=EMBED_DIM, num_heads=4, num_layers=3):

        super().__init__()

        if img_size % patch_size:

            raise ValueError("IMG_SIZE must be divisible by PATCH_SIZE.")

        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(

            1, embed_dim, kernel_size=patch_size, stride=patch_size

        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,

            dropout=0.1, batch_first=True, norm_first=True, activation="gelu"

        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        self.cosine_classifier = CosineClassifier(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)



    def forward_features(self, x):

        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(x.size(0), -1, -1)

        x = torch.cat([cls, x], dim=1)

        x = self.encoder(x + self.pos_embed)

        return self.norm(x[:, 0])



    def forward(self, x):

        return self.cosine_classifier(self.forward_features(x))



# -------------------------

# F. Class weights and training

# -------------------------

train_class_ids = encoded_labels[train_idx]

class_weight_values = compute_class_weight(

    class_weight="balanced",

    classes=np.arange(num_classes),

    y=train_class_ids

)

class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=DEVICE)



logger.info("\nClass weights:")

for name, weight in zip(class_names, class_weight_values):

    logger.info(f"{name}: {weight:.3f}")



def train_model(model, model_name):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6

    )



    history = []

    best_val_loss = float("inf")

    best_state = None

    bad_epochs = 0

    checkpoint_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_cosine_best.pt")



    for epoch in range(1, EPOCHS + 1):

        t0 = time.time()

        model.train()

        total_loss, total_correct, total_n = 0.0, 0, 0



        for images, labels_batch in train_loader:

            images = images.to(DEVICE, non_blocking=True)

            labels_batch = labels_batch.to(DEVICE, non_blocking=True)



            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = criterion(logits, labels_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()



            total_loss += loss.item() * labels_batch.size(0)

            total_correct += (logits.argmax(1) == labels_batch).sum().item()

            total_n += labels_batch.size(0)



        train_loss = total_loss / max(total_n, 1)

        train_acc = total_correct / max(total_n, 1)



        model.eval()

        val_loss_sum, val_correct, val_n = 0.0, 0, 0

        with torch.no_grad():

            for images, labels_batch in val_loader:

                images = images.to(DEVICE, non_blocking=True)

                labels_batch = labels_batch.to(DEVICE, non_blocking=True)

                logits = model(images)

                loss = criterion(logits, labels_batch)

                val_loss_sum += loss.item() * labels_batch.size(0)

                val_correct += (logits.argmax(1) == labels_batch).sum().item()

                val_n += labels_batch.size(0)



        val_loss = val_loss_sum / max(val_n, 1)

        val_acc = val_correct / max(val_n, 1)

        scheduler.step(val_loss)



        history.append({

            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,

            "val_loss": val_loss, "val_accuracy": val_acc,

            "learning_rate": optimizer.param_groups[0]["lr"]

        })

        logger.info(

            f"{model_name} | Epoch {epoch:03d}/{EPOCHS} | "

            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "

            f"val loss {val_loss:.4f}, acc {val_acc:.4f} | "

            f"{time.time()-t0:.1f}s"

        )



        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            torch.save({

                "model_name": model_name,

                "state_dict": best_state,

                "class_names": class_names,

                "img_size": IMG_SIZE,

                "patch_size": PATCH_SIZE,

                "embedding_dim": EMBED_DIM

            }, checkpoint_path)

            bad_epochs = 0

        else:

            bad_epochs += 1

            if bad_epochs >= EARLY_STOPPING_PATIENCE:

                logger.info(f"Early stopping {model_name} at epoch {epoch}.")

                break



    if best_state is not None:

        model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(

        os.path.join(OUTPUT_DIR, f"{model_name.lower()}_training_history.csv"), index=False

    )

    return model, history, checkpoint_path



# -------------------------

# G. Extract embeddings and cosine class prototypes

# -------------------------

@torch.no_grad()

def extract_embeddings(model, loader):

    model.eval()

    all_embeddings, all_labels = [], []

    for images, labels_batch in loader:

        images = images.to(DEVICE, non_blocking=True)

        embeddings = model.forward_features(images)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

        all_labels.append(labels_batch.cpu())

    return torch.cat(all_embeddings).numpy(), torch.cat(all_labels).numpy()



def build_class_prototypes(train_embeddings, train_targets):

    """Mean training embedding for each class, L2-normalized."""

    prototypes = []

    for class_id in range(num_classes):

        class_features = train_embeddings[train_targets == class_id]

        if len(class_features) == 0:

            raise ValueError(f"No training embeddings found for class {class_names[class_id]}")

        centroid = class_features.mean(axis=0)

        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        prototypes.append(centroid)

    return np.stack(prototypes, axis=0).astype(np.float32)



def cosine_predict(embeddings, prototypes):

    # Both inputs are normalized. Matrix product gives cosine similarity to every class.

    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12)

    similarities = embeddings @ prototypes.T

    predictions = similarities.argmax(axis=1)

    return predictions, similarities



def evaluate_cosine_prototypes(model, model_name, train_embed, train_y):

    prototypes = build_class_prototypes(train_embed, train_y)

    test_embeddings, test_y = extract_embeddings(model, test_loader)

    test_pred, test_sim = cosine_predict(test_embeddings, prototypes)



    accuracy = accuracy_score(test_y, test_pred)

    precision_macro = precision_score(test_y, test_pred, average="macro", zero_division=0)

    recall_macro = recall_score(test_y, test_pred, average="macro", zero_division=0)

    f1_macro = f1_score(test_y, test_pred, average="macro", zero_division=0)

    f1_weighted = f1_score(test_y, test_pred, average="weighted", zero_division=0)



    report = classification_report(

        test_y, test_pred, labels=list(range(num_classes)),

        target_names=class_names, zero_division=0

    )

    cm = confusion_matrix(test_y, test_pred, labels=list(range(num_classes)))



    with open(os.path.join(OUTPUT_DIR, f"{model_name.lower()}_cosine_classification_report.txt"), "w") as f:

        f.write(report)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(

        os.path.join(OUTPUT_DIR, f"{model_name.lower()}_cosine_confusion_matrix.csv")

    )

    pd.DataFrame({

        "true_label": [idx_to_label[int(y)] for y in test_y],

        "predicted_label": [idx_to_label[int(y)] for y in test_pred],

        "max_cosine_similarity": test_sim.max(axis=1)

    }).to_csv(os.path.join(OUTPUT_DIR, f"{model_name.lower()}_cosine_test_predictions.csv"), index=False)



    # Save embeddings, prototypes, and full similarity matrix for later analysis.

    np.save(os.path.join(OUTPUT_DIR, f"{model_name.lower()}_train_embeddings.npy"), train_embed)

    np.save(os.path.join(OUTPUT_DIR, f"{model_name.lower()}_class_prototypes.npy"), prototypes)

    np.save(os.path.join(OUTPUT_DIR, f"{model_name.lower()}_test_cosine_similarities.npy"), test_sim)



    # Confusion matrix figure

    fig, ax = plt.subplots(figsize=(max(9, num_classes * 0.8), max(7, num_classes * 0.65)))

    im = ax.imshow(cm, interpolation="nearest")

    ax.set_title(f"{model_name} — Test Confusion Matrix (Cosine Prototypes)")

    ax.set_xlabel("Predicted class")

    ax.set_ylabel("True class")

    ax.set_xticks(range(num_classes))

    ax.set_yticks(range(num_classes))

    ax.set_xticklabels(class_names, rotation=90)

    ax.set_yticklabels(class_names)

    fig.colorbar(im, ax=ax)

    fig.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, f"{model_name.lower()}_cosine_confusion_matrix.png"), dpi=160)

    plt.close(fig)



    logger.info(f"\n{model_name} — test results using cosine similarity to class prototypes")

    logger.info(f"Accuracy:     {accuracy:.4f}")

    logger.info(f"Macro precision: {precision_macro:.4f}")

    logger.info(f"Macro recall:    {recall_macro:.4f}")

    logger.info(f"Macro F1:        {f1_macro:.4f}")

    logger.info(f"Weighted F1:     {f1_weighted:.4f}")

    logger.info(report)



    return {

        "model": model_name,

        "accuracy": accuracy,

        "precision_macro": precision_macro,

        "recall_macro": recall_macro,

        "f1_macro": f1_macro,

        "f1_weighted": f1_weighted

    }



# -------------------------

# H. Train all three models and compare cosine prototype results

# -------------------------

model_factories = {

    "CNN": lambda: SEMCNN(num_classes),

    "LSTM": lambda: SEMLSTM(num_classes),

    "Transformer": lambda: SEMTransformer(num_classes)

}



summary_rows = []

trained_models = {}



for model_name, factory in model_factories.items():

    logger.info("\n" + "=" * 76)

    logger.info(f"Training: {model_name}")

    logger.info("=" * 76)

    model, history, checkpoint_path = train_model(factory(), model_name)

    trained_models[model_name] = model



    # Build prototypes from non-augmented TRAIN embeddings only.

    train_embeddings, train_targets = extract_embeddings(model, train_eval_loader)

    result = evaluate_cosine_prototypes(model, model_name, train_embeddings, train_targets)

    result["epochs_trained"] = len(history)

    result["best_model_path"] = checkpoint_path

    summary_rows.append(result)



summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro", ascending=False)

summary_csv = os.path.join(OUTPUT_DIR, "cosine_model_comparison.csv")

summary_df.to_csv(summary_csv, index=False)



with open(os.path.join(OUTPUT_DIR, "cosine_model_comparison.json"), "w") as f:

    json.dump(summary_rows, f, indent=2)



logger.info("\n" + "=" * 76)

logger.info("FINAL COMPARISON — TEST SET / COSINE PROTOTYPE CLASSIFIER")

logger.info("=" * 76)

logger.info(summary_df.to_string(index=False))

logger.info(f"\nResults saved to: {summary_csv}")



ax = summary_df.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]].plot(

    kind="bar", figsize=(12, 6), ylim=(0, 1)

)

ax.set_title("CNN vs LSTM vs Transformer — Cosine Similarity Results")

ax.set_ylabel("Score")

ax.set_xlabel("Model")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "cosine_model_comparison.png"), dpi=160)

plt.show()



logger.info(f"\nAll outputs saved under: {OUTPUT_DIR}")

for filename in sorted(os.listdir(OUTPUT_DIR)):

    logger.info(f" - {filename}")
