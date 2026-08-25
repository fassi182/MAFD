from pathlib import Path
from collections import defaultdict
import hashlib
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 2. Dataset paths
# ============================================================

DATASET_ROOT = Path("/content/WBCAtt")

ANNOTATION_ROOT = DATASET_ROOT / "annotations"
IMAGE_ROOT = DATASET_ROOT / "PBC_dataset_normal_DIB"


ORIGINAL_CSV_PATHS = {
    "train": (
        ANNOTATION_ROOT
        / "pbc_attr_v1_train.csv"
    ),

    "validation": (
        ANNOTATION_ROOT
        / "pbc_attr_v1_val.csv"
    ),

    "test": (
        ANNOTATION_ROOT
        / "test.csv"
    ),
}


if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset folder was not found:\n{DATASET_ROOT}"
    )


if not ANNOTATION_ROOT.exists():
    raise FileNotFoundError(
        f"Annotation folder was not found:\n"
        f"{ANNOTATION_ROOT}"
    )


if not IMAGE_ROOT.exists():
    raise FileNotFoundError(
        f"Image folder was not found:\n{IMAGE_ROOT}"
    )


# ============================================================
# 3. Output directory
# ============================================================

GOOGLE_DRIVE_OUTPUT = Path(
    "/content/drive/MyDrive/MAFD_OUTPUTS/model_a_clean"
)

LOCAL_OUTPUT = Path(
    "/content/MAFD_OUTPUTS/model_a_clean"
)

if Path("/content/drive/MyDrive").exists():
    OUTPUT_DIR = GOOGLE_DRIVE_OUTPUT
else:
    OUTPUT_DIR = LOCAL_OUTPUT

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLEAN_ANNOTATION_DIR = (
    OUTPUT_DIR / "clean_annotations"
)

CLEAN_ANNOTATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REMOVAL_LOG_PATH = (
    OUTPUT_DIR
    / "exact_duplicate_removal_log.csv"
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "model_a_clean_resnet18_best.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "model_a_clean_training_history.csv"
)

TEST_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "model_a_clean_test_predictions.csv"
)

TEST_REPORT_PATH = (
    OUTPUT_DIR
    / "model_a_clean_test_classification_report.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "model_a_clean_test_confusion_matrix.png"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "model_a_clean_config.json"
)


# ============================================================
# 4. Dataset definitions
# ============================================================

LABEL_COLUMN = "label"
IMAGE_COLUMN = "path"

CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


ATTRIBUTE_COLUMNS = [
    "cell_size",
    "cell_shape",
    "nucleus_shape",
    "nuclear_cytoplasmic_ratio",
    "chromatin_density",
    "cytoplasm_vacuole",
    "cytoplasm_texture",
    "cytoplasm_colour",
    "granule_type",
    "granule_colour",
    "granularity",
]


ATTRIBUTE_VALUES = {
    "cell_size": [
        "big",
        "small",
    ],

    "cell_shape": [
        "irregular",
        "round",
    ],

    "nucleus_shape": [
        "irregular",
        "segmented-bilobed",
        "segmented-multilobed",
        "unsegmented-band",
        "unsegmented-indented",
        "unsegmented-round",
    ],

    "nuclear_cytoplasmic_ratio": [
        "high",
        "low",
    ],

    "chromatin_density": [
        "densely",
        "loosely",
    ],

    "cytoplasm_vacuole": [
        "no",
        "yes",
    ],

    "cytoplasm_texture": [
        "clear",
        "frosted",
    ],

    "cytoplasm_colour": [
        "blue",
        "light blue",
        "purple blue",
    ],

    "granule_type": [
        "coarse",
        "nil",
        "round",
        "small",
    ],

    "granule_colour": [
        "nil",
        "pink",
        "purple",
        "red",
    ],

    "granularity": [
        "no",
        "yes",
    ],
}


ATTRIBUTE_TO_INDEX = {
    attribute_name: {
        value: index
        for index, value in enumerate(values)
    }
    for attribute_name, values in ATTRIBUTE_VALUES.items()
}


# ============================================================
# 5. Image-path resolution
# ============================================================

def resolve_image_path(raw_value, label):
    """
    Resolves images from absolute paths, relative paths,
    filenames, and class-folder paths.
    """

    value = str(raw_value).strip()
    value = value.replace("\\", "/")

    raw_path = Path(value)

    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)

    candidates.extend(
        [
            raw_path,
            DATASET_ROOT / raw_path,
            IMAGE_ROOT / raw_path,
            IMAGE_ROOT / label / raw_path,
            IMAGE_ROOT / label / raw_path.name,
            IMAGE_ROOT / raw_path.name,
            DATASET_ROOT / raw_path.name,
        ]
    )

    checked_paths = set()

    for candidate in candidates:

        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        if candidate in checked_paths:
            continue

        checked_paths.add(candidate)

        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not resolve image path.\n"
        f"CSV value: {raw_value}\n"
        f"Label: {label}"
    )


# ============================================================
# 6. Exact file hashing
# ============================================================

def calculate_sha256(image_path):
    """
    Calculates the exact binary-content hash of an image.
    """

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as image_file:

        for chunk in iter(
            lambda: image_file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 7. Load original CSV files and calculate hashes
# ============================================================

def load_and_hash_split(split_name, csv_path):
    """
    Loads one split and calculates a SHA-256 hash for
    every referenced image.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file was not found:\n{csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = [
        LABEL_COLUMN,
        IMAGE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {split_name} CSV:\n"
            f"{missing_columns}"
        )

    records = []

    total_rows = len(dataframe)

    for row_number, row in dataframe.iterrows():

        label = str(
            row[LABEL_COLUMN]
        ).strip().lower()

        if label not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unexpected class label in {split_name}, "
                f"row {row_number}: {label}"
            )

        image_path = resolve_image_path(
            raw_value=row[IMAGE_COLUMN],
            label=label,
        )

        file_hash = calculate_sha256(
            image_path
        )

        records.append(
            {
                "split": split_name,
                "row_number": row_number,
                "label": label,
                "resolved_path": str(image_path),
                "sha256": file_hash,
            }
        )

        print(
            f"Hashing {split_name}: "
            f"{row_number + 1}/{total_rows}",
            end="\r",
        )

    print()

    return dataframe, records


# ============================================================
# 8. Remove exact duplicates from lower-priority splits
# ============================================================

def create_clean_splits():
    """
    Preserves the duplicate in the highest-priority split.

    Priority:
        test > validation > train

    Therefore:
        test copies are preserved,
        validation copies are preserved over train,
        train copies are removed when duplicated.
    """

    print("=" * 70)
    print("CREATING CLEAN DATASET SPLITS")
    print("=" * 70)

    all_dataframes = {}
    all_records = []

    for split_name, csv_path in ORIGINAL_CSV_PATHS.items():

        print()
        print(
            f"Loading and hashing {split_name}"
        )

        dataframe, records = load_and_hash_split(
            split_name=split_name,
            csv_path=csv_path,
        )

        all_dataframes[split_name] = dataframe
        all_records.extend(records)

    split_priority = {
        "train": 1,
        "validation": 2,
        "test": 3,
    }

    hash_groups = defaultdict(list)

    for record in all_records:
        hash_groups[
            record["sha256"]
        ].append(record)

    rows_to_remove = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    removal_log = []
    duplicate_group_number = 0

    for file_hash, records in hash_groups.items():

        unique_splits = set(
            record["split"]
            for record in records
        )

        if len(unique_splits) <= 1:
            continue

        duplicate_group_number += 1

        records_sorted = sorted(
            records,
            key=lambda record: split_priority[
                record["split"]
            ],
            reverse=True,
        )

        kept_record = records_sorted[0]

        for duplicate_record in records_sorted[1:]:

            rows_to_remove[
                duplicate_record["split"]
            ].add(
                duplicate_record["row_number"]
            )

            removal_log.append(
                {
                    "duplicate_group": (
                        duplicate_group_number
                    ),
                    "sha256": file_hash,
                    "kept_split": (
                        kept_record["split"]
                    ),
                    "kept_row": (
                        kept_record["row_number"]
                    ),
                    "kept_label": (
                        kept_record["label"]
                    ),
                    "kept_path": (
                        kept_record["resolved_path"]
                    ),
                    "removed_split": (
                        duplicate_record["split"]
                    ),
                    "removed_row": (
                        duplicate_record["row_number"]
                    ),
                    "removed_label": (
                        duplicate_record["label"]
                    ),
                    "removed_path": (
                        duplicate_record["resolved_path"]
                    ),
                    "reason": (
                        "Exact duplicate removed "
                        "from lower-priority split"
                    ),
                }
            )

    removal_log_dataframe = pd.DataFrame(
        removal_log
    )

    removal_log_dataframe.to_csv(
        REMOVAL_LOG_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Exact duplicate rows removed: "
        f"{len(removal_log)}"
    )

    print(
        f"Removal log saved to:\n"
        f"{REMOVAL_LOG_PATH}"
    )

    clean_csv_paths = {}

    for split_name, dataframe in all_dataframes.items():

        rows_to_delete = rows_to_remove[
            split_name
        ]

        clean_dataframe = dataframe[
            ~dataframe.index.isin(rows_to_delete)
        ].copy()

        if split_name == "train":
            output_filename = (
                "pbc_attr_v1_train_clean.csv"
            )

        elif split_name == "validation":
            output_filename = (
                "pbc_attr_v1_val_clean.csv"
            )

        else:
            output_filename = "test_clean.csv"

        output_path = (
            CLEAN_ANNOTATION_DIR
            / output_filename
        )

        clean_dataframe.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        clean_csv_paths[split_name] = output_path

        print()
        print(
            f"{split_name.capitalize()} clean CSV saved:"
        )
        print(f"Path: {output_path}")
        print(f"Original rows: {len(dataframe)}")
        print(
            f"Removed rows: "
            f"{len(rows_to_delete)}"
        )
        print(
            f"Remaining rows: "
            f"{len(clean_dataframe)}"
        )

    return clean_csv_paths


# ============================================================
# 9. Verify that no exact duplicates remain
# ============================================================

def verify_clean_splits(clean_csv_paths):
    """
    Verifies that the cleaned CSV files contain no
    exact duplicate images across splits.
    """

    print()
    print("=" * 70)
    print("VERIFYING CLEANED SPLITS")
    print("=" * 70)

    hash_to_splits = defaultdict(set)

    for split_name, csv_path in clean_csv_paths.items():

        dataframe = pd.read_csv(csv_path)

        for row_number, row in dataframe.iterrows():

            label = str(
                row[LABEL_COLUMN]
            ).strip().lower()

            image_path = resolve_image_path(
                raw_value=row[IMAGE_COLUMN],
                label=label,
            )

            file_hash = calculate_sha256(
                image_path
            )

            hash_to_splits[
                file_hash
            ].add(split_name)

    remaining_duplicate_hashes = [
        file_hash
        for file_hash, split_names
        in hash_to_splits.items()
        if len(split_names) > 1
    ]

    print(
        "Exact duplicate hashes remaining: "
        f"{len(remaining_duplicate_hashes)}"
    )

    if len(remaining_duplicate_hashes) != 0:
        raise RuntimeError(
            "Exact duplicates still remain across "
            "the cleaned splits."
        )

    print(
        "Clean split verification: PASSED"
    )


# ============================================================
# 10. Image transformations
# ============================================================

IMAGE_SIZE = 224

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


evaluation_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# 11. Dataset class
# ============================================================

class WBCAttCleanDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split_name,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.split_name = split_name
        self.transform = transform

        self.dataframe = pd.read_csv(
            self.csv_path
        )

        self.image_paths = []
        self.labels = []
        self.attributes = []

        for row_number, row in self.dataframe.iterrows():

            label = str(
                row[LABEL_COLUMN]
            ).strip().lower()

            if label not in CLASS_TO_INDEX:
                raise ValueError(
                    f"Invalid label in {split_name}, "
                    f"row {row_number}: {label}"
                )

            image_path = resolve_image_path(
                raw_value=row[IMAGE_COLUMN],
                label=label,
            )

            attribute_vector = []

            for attribute_name in ATTRIBUTE_COLUMNS:

                value = str(
                    row[attribute_name]
                ).strip().lower()

                valid_values = ATTRIBUTE_TO_INDEX[
                    attribute_name
                ]

                if value not in valid_values:
                    raise ValueError(
                        f"Invalid attribute value.\n"
                        f"Split: {split_name}\n"
                        f"Row: {row_number}\n"
                        f"Attribute: {attribute_name}\n"
                        f"Value: {value}"
                    )

                attribute_vector.append(
                    valid_values[value]
                )

            self.image_paths.append(
                image_path
            )

            self.labels.append(
                CLASS_TO_INDEX[label]
            )

            self.attributes.append(
                attribute_vector
            )

        self.labels = torch.tensor(
            self.labels,
            dtype=torch.long,
        )

        self.attributes = torch.tensor(
            self.attributes,
            dtype=torch.long,
        )

        print()
        print(
            f"{split_name.capitalize()} clean dataset loaded"
        )
        print(
            f"Samples: {len(self.image_paths)}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]

        attribute_vector = self.attributes[
            index
        ]

        return (
            image,
            label,
            attribute_vector,
            str(image_path),
        )


# ============================================================
# 12. Load cleaned datasets
# ============================================================

clean_csv_paths = create_clean_splits()

verify_clean_splits(
    clean_csv_paths
)

train_dataset = WBCAttCleanDataset(
    csv_path=clean_csv_paths["train"],
    split_name="train",
    transform=train_transform,
)

validation_dataset = WBCAttCleanDataset(
    csv_path=clean_csv_paths["validation"],
    split_name="validation",
    transform=evaluation_transform,
)

test_dataset = WBCAttCleanDataset(
    csv_path=clean_csv_paths["test"],
    split_name="test",
    transform=evaluation_transform,
)


# ============================================================
# 13. DataLoaders
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 14. Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 70)
print("MODEL A CLEAN TRAINING")
print("=" * 70)

print()
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()
print(
    f"Clean train samples: "
    f"{len(train_dataset)}"
)

print(
    f"Clean validation samples: "
    f"{len(validation_dataset)}"
)

print(
    f"Clean test samples: "
    f"{len(test_dataset)}"
)


# ============================================================
# 15. Calculate class weights
# ============================================================

train_class_counts = torch.bincount(
    train_dataset.labels,
    minlength=len(CLASS_NAMES),
).float()

total_train_samples = (
    train_class_counts.sum()
)

class_weights = total_train_samples / (
    len(CLASS_NAMES)
    * train_class_counts
)

class_weights = class_weights.to(DEVICE)

print()
print("=" * 70)
print("CLEAN TRAIN CLASS DISTRIBUTION")
print("=" * 70)

for class_index, class_name in enumerate(
    CLASS_NAMES
):
    print(
        f"{class_name:15s} "
        f"count="
        f"{int(train_class_counts[class_index])} "
        f"weight="
        f"{class_weights[class_index].item():.4f}"
    )


# ============================================================
# 16. Build ResNet18 Model A
# ============================================================

USE_PRETRAINED_WEIGHTS = True


def build_model():
    if USE_PRETRAINED_WEIGHTS:

        try:
            print()
            print(
                "Loading pretrained ResNet18 weights..."
            )

            model = resnet18(
                weights=ResNet18_Weights.DEFAULT
            )

            print(
                "Pretrained weights loaded."
            )

        except Exception as error:

            print()
            print(
                "Could not load pretrained weights."
            )

            print(
                "Using randomly initialized weights."
            )

            print(
                f"Reason: {error}"
            )

            model = resnet18(
                weights=None
            )

    else:
        model = resnet18(
            weights=None
        )

    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features=number_of_features,
        out_features=len(CLASS_NAMES),
    )

    return model


model = build_model()
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0001,
)

scheduler = ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# 17. Metrics
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
):
    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            true_labels,
            predicted_labels,
        ),

        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),

        "macro_precision": precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),

        "macro_recall": recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# 18. One training or evaluation epoch
# ============================================================

def run_one_epoch(
    model,
    data_loader,
    criterion,
    device,
    optimizer=None,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels, _, _ in data_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_training
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            if is_training:

                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_probabilities.append(
            probabilities.detach()
            .cpu()
        )

    average_loss = (
        total_loss / total_samples
    )

    all_labels = np.asarray(
        all_labels
    )

    all_predictions = np.asarray(
        all_predictions
    )

    all_probabilities = torch.cat(
        all_probabilities,
        dim=0,
    ).numpy()

    metrics = calculate_metrics(
        true_labels=all_labels,
        predicted_labels=all_predictions,
    )

    metrics["loss"] = average_loss

    return (
        metrics,
        all_labels,
        all_predictions,
        all_probabilities,
    )


# ============================================================
# 19. Train Model A
# ============================================================

NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5

history = []

best_validation_macro_f1 = -1.0
epochs_without_improvement = 0

print()
print("=" * 70)
print("STARTING CLEAN MODEL A TRAINING")
print("=" * 70)

for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    print()
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )
    print("-" * 70)

    train_metrics, _, _, _ = run_one_epoch(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        device=DEVICE,
        optimizer=optimizer,
    )

    validation_metrics, _, _, _ = run_one_epoch(
        model=model,
        data_loader=validation_loader,
        criterion=criterion,
        device=DEVICE,
        optimizer=None,
    )

    scheduler.step(
        validation_metrics["loss"]
    )

    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"Train loss:              "
        f"{train_metrics['loss']:.4f}"
    )

    print(
        f"Train accuracy:          "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Train balanced accuracy: "
        f"{train_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Train macro F1:          "
        f"{train_metrics['macro_f1']:.4f}"
    )

    print(
        f"Validation loss:         "
        f"{validation_metrics['loss']:.4f}"
    )

    print(
        f"Validation accuracy:     "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation balanced acc: "
        f"{validation_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Validation macro F1:     "
        f"{validation_metrics['macro_f1']:.4f}"
    )

    print(
        f"Learning rate:           "
        f"{current_learning_rate:.8f}"
    )

    history.append(
        {
            "epoch": epoch,
            "learning_rate": (
                current_learning_rate
            ),

            "train_loss": (
                train_metrics["loss"]
            ),

            "train_accuracy": (
                train_metrics["accuracy"]
            ),

            "train_balanced_accuracy": (
                train_metrics[
                    "balanced_accuracy"
                ]
            ),

            "train_macro_f1": (
                train_metrics["macro_f1"]
            ),

            "validation_loss": (
                validation_metrics["loss"]
            ),

            "validation_accuracy": (
                validation_metrics[
                    "accuracy"
                ]
            ),

            "validation_balanced_accuracy": (
                validation_metrics[
                    "balanced_accuracy"
                ]
            ),

            "validation_macro_f1": (
                validation_metrics[
                    "macro_f1"
                ]
            ),
        }
    )

    if (
        validation_metrics["macro_f1"]
        > best_validation_macro_f1
    ):

        best_validation_macro_f1 = (
            validation_metrics["macro_f1"]
        )

        epochs_without_improvement = 0

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "best_validation_macro_f1": (
                best_validation_macro_f1
            ),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "seed": SEED,
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH,
        )

        print()
        print(
            "Best clean model checkpoint saved."
        )

    else:

        epochs_without_improvement += 1

        print()
        print(
            "No validation improvement."
        )

        print(
            f"Patience: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()
        print(
            "Early stopping activated."
        )

        break


# ============================================================
# 20. Save training history
# ============================================================

history_dataframe = pd.DataFrame(
    history
)

history_dataframe.to_csv(
    HISTORY_PATH,
    index=False,
)

print()
print(
    f"Training history saved to:\n"
    f"{HISTORY_PATH}"
)


# ============================================================
# 21. Load best clean checkpoint
# ============================================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print()
print("=" * 70)
print("BEST CLEAN MODEL LOADED")
print("=" * 70)

print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation macro F1: "
    f"{checkpoint['best_validation_macro_f1']:.4f}"
)


# ============================================================
# 22. Final clean test evaluation
# ============================================================

test_metrics, test_labels, test_predictions, test_probabilities = (
    run_one_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        optimizer=None,
    )
)

print()
print("=" * 70)
print("FINAL CLEAN TEST RESULTS")
print("=" * 70)

print(
    f"Test loss:              "
    f"{test_metrics['loss']:.4f}"
)

print(
    f"Test accuracy:          "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test balanced accuracy: "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Test macro F1:          "
    f"{test_metrics['macro_f1']:.4f}"
)

print(
    f"Test weighted F1:       "
    f"{test_metrics['weighted_f1']:.4f}"
)

print(
    f"Test macro precision:   "
    f"{test_metrics['macro_precision']:.4f}"
)

print(
    f"Test macro recall:      "
    f"{test_metrics['macro_recall']:.4f}"
)


# ============================================================
# 23. Per-class test report
# ============================================================

classification_report_text = classification_report(
    y_true=test_labels,
    y_pred=test_predictions,
    target_names=CLASS_NAMES,
    zero_division=0,
)

print()
print("=" * 70)
print("PER-CLASS CLEAN TEST RESULTS")
print("=" * 70)
print()
print(classification_report_text)

classification_report_dictionary = classification_report(
    y_true=test_labels,
    y_pred=test_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

classification_report_dataframe = pd.DataFrame(
    classification_report_dictionary
).transpose()

classification_report_dataframe.to_csv(
    TEST_REPORT_PATH,
    index=True,
)


# ============================================================
# 24. Save test predictions and confidence
# ============================================================

test_image_paths = [
    str(path)
    for path in test_dataset.image_paths
]

test_confidence = test_probabilities.max(
    axis=1
)

prediction_dataframe = pd.DataFrame(
    {
        "image_path": test_image_paths,

        "true_label_index": test_labels,

        "true_label": [
            CLASS_NAMES[index]
            for index in test_labels
        ],

        "predicted_label_index": (
            test_predictions
        ),

        "predicted_label": [
            CLASS_NAMES[index]
            for index in test_predictions
        ],

        "confidence": test_confidence,

        "correct": (
            test_labels == test_predictions
        ),
    }
)

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    probability_column = (
        f"probability_{class_name}"
    )

    prediction_dataframe[
        probability_column
    ] = test_probabilities[
        :,
        class_index
    ]

prediction_dataframe.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False,
)

print()
print(
    f"Test predictions saved to:\n"
    f"{TEST_PREDICTIONS_PATH}"
)


# ============================================================
# 25. Create confusion matrix
# ============================================================

confusion = confusion_matrix(
    y_true=test_labels,
    y_pred=test_predictions,
    labels=list(range(len(CLASS_NAMES))),
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)

plt.title(
    "Model A Clean Test Confusion Matrix"
)

plt.xlabel(
    "Predicted label"
)

plt.ylabel(
    "True label"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
)

plt.close()

print(
    f"Confusion matrix saved to:\n"
    f"{CONFUSION_MATRIX_PATH}"
)


# ============================================================
# 26. Save configuration
# ============================================================

configuration = {
    "experiment": (
        "Model A after exact duplicate removal"
    ),
    "seed": SEED,
    "device": str(DEVICE),
    "architecture": "ResNet18",
    "pretrained_weights": (
        USE_PRETRAINED_WEIGHTS
    ),
    "batch_size": BATCH_SIZE,
    "num_epochs_requested": NUM_EPOCHS,
    "image_size": IMAGE_SIZE,
    "number_of_classes": len(CLASS_NAMES),
    "class_names": CLASS_NAMES,
    "train_samples": len(train_dataset),
    "validation_samples": len(
        validation_dataset
    ),
    "test_samples": len(test_dataset),
    "best_epoch": int(
        checkpoint["epoch"]
    ),
    "best_validation_macro_f1": float(
        checkpoint[
            "best_validation_macro_f1"
        ]
    ),
    "test_loss": float(
        test_metrics["loss"]
    ),
    "test_accuracy": float(
        test_metrics["accuracy"]
    ),
    "test_balanced_accuracy": float(
        test_metrics[
            "balanced_accuracy"
        ]
    ),
    "test_macro_f1": float(
        test_metrics["macro_f1"]
    ),
    "test_weighted_f1": float(
        test_metrics["weighted_f1"]
    ),
    "test_macro_precision": float(
        test_metrics[
            "macro_precision"
        ]
    ),
    "test_macro_recall": float(
        test_metrics["macro_recall"]
    ),
    "exact_duplicate_rows_removed": 6,
}

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8",
) as configuration_file:

    json.dump(
        configuration,
        configuration_file,
        indent=4,
    )


# ============================================================
# 27. Final message
# ============================================================

print()
print("=" * 70)
print("CLEAN MODEL A TRAINING FINISHED SUCCESSFULLY")
print("=" * 70)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Generated files:")

print(BEST_MODEL_PATH)
print(HISTORY_PATH)
print(TEST_PREDICTIONS_PATH)
print(TEST_REPORT_PATH)
print(CONFUSION_MATRIX_PATH)
print(CONFIG_PATH)
print(REMOVAL_LOG_PATH)

print()
print(
    "The clean test result is the result to use "
    "in the research paper."
)
