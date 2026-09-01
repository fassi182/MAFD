

from pathlib import Path
import copy
import importlib.util
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW, LBFGS
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ============================================================
# 1. Reproducibility
# ============================================================

SEEDS = [
    7,
    21,
    42,
    84,
    123,
]

BATCH_SIZE = 32
MAX_EPOCHS = 15
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 5

NUM_CLASSES = 5

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Paths
# ============================================================

MODEL_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/MAFD_OUTPUTS/model_a_clean"
)

FALLBACK_OUTPUT_DIR = Path(
    "/content/MAFD_OUTPUTS/model_a_clean"
)

if not MODEL_OUTPUT_DIR.exists():
    MODEL_OUTPUT_DIR = FALLBACK_OUTPUT_DIR

CLEAN_ANNOTATION_DIR = (
    MODEL_OUTPUT_DIR
    / "clean_annotations"
)

TRAIN_CSV = (
    CLEAN_ANNOTATION_DIR
    / "pbc_attr_v1_train_clean.csv"
)

VALIDATION_CSV = (
    CLEAN_ANNOTATION_DIR
    / "pbc_attr_v1_val_clean.csv"
)

TEST_CSV = (
    CLEAN_ANNOTATION_DIR
    / "test_clean.csv"
)

OUTPUT_DIR = (
    MODEL_OUTPUT_DIR
    / "repeated_seed_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "repeated_seed_summary.csv"
)

TARGET_GROUP_PATH = (
    OUTPUT_DIR
    / "repeated_seed_target_group_predictions.csv"
)

PLOT_PATH = (
    OUTPUT_DIR
    / "repeated_seed_accuracy_ece.png"
)


# ============================================================
# 3. Check files
# ============================================================

required_files = [
    TRAIN_CSV,
    VALIDATION_CSV,
    TEST_CSV,
]

for required_file in required_files:

    if not required_file.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n"
            f"{required_file}"
        )


# ============================================================
# 4. Load the existing Dataset class
# ============================================================

def load_dataset_module():

    possible_files = [
        Path("/content/2_test_dataloader.py"),
        Path("/content/02_test_dataloader.py"),
        Path("2_test_dataloader.py"),
        Path("02_test_dataloader.py"),
    ]

    dataset_file = None

    for file_path in possible_files:

        if file_path.exists():
            dataset_file = file_path
            break

    if dataset_file is None:
        raise FileNotFoundError(
            "Could not find the previous DataLoader file.\n"
            "Expected one of:\n"
            f"{possible_files}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            "wbc_dataset_module_repeated_seed",
            dataset_file,
        )
    )

    if specification is None:
        raise ImportError(
            "Could not create the Dataset module."
        )

    dataset_module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    if specification.loader is None:
        raise ImportError(
            "Could not load the Dataset module."
        )

    specification.loader.exec_module(
        dataset_module
    )

    return dataset_module


dataset_module = load_dataset_module()

WBCAttDataset = (
    dataset_module.WBCAttDataset
)

evaluation_transform = (
    dataset_module.evaluation_transform
)

training_transform = getattr(
    dataset_module,
    "training_transform",
    evaluation_transform,
)


# ============================================================
# 5. Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("REPEATED-SEED VALIDATION")
print("=" * 70)

print()
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# 6. Load metadata
# ============================================================

train_dataframe = pd.read_csv(
    TRAIN_CSV
)

validation_dataframe = pd.read_csv(
    VALIDATION_CSV
)

test_dataframe = pd.read_csv(
    TEST_CSV
)

for dataframe in [
    train_dataframe,
    validation_dataframe,
    test_dataframe,
]:

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# 7. Calibration functions
# ============================================================

def calculate_ece(
    confidence,
    correctness,
    number_of_bins=10,
):

    confidence = np.asarray(
        confidence,
        dtype=float,
    )

    correctness = np.asarray(
        correctness,
        dtype=float,
    )

    if len(confidence) == 0:
        return np.nan

    ece = 0.0

    for bin_index in range(number_of_bins):

        lower = (
            bin_index
            / number_of_bins
        )

        upper = (
            bin_index + 1
        ) / number_of_bins

        if bin_index == number_of_bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        if mask.sum() == 0:
            continue

        bin_accuracy = (
            correctness[mask].mean()
        )

        bin_confidence = (
            confidence[mask].mean()
        )

        bin_fraction = (
            mask.sum()
            / len(confidence)
        )

        ece += (
            bin_fraction
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(ece)


def fit_temperature(
    validation_logits,
    validation_labels,
):

    validation_logits = (
        validation_logits.to(DEVICE)
    )

    validation_labels = (
        validation_labels.to(DEVICE)
    )

    log_temperature = nn.Parameter(
        torch.zeros(
            1,
            device=DEVICE,
        )
    )

    optimizer = LBFGS(
        [
            log_temperature
        ],
        lr=0.01,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    criterion = nn.CrossEntropyLoss()

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        scaled_logits = (
            validation_logits
            / temperature
        )

        loss = criterion(
            scaled_logits,
            validation_labels,
        )

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    temperature = torch.exp(
        log_temperature
    ).detach().item()

    return float(temperature)


# ============================================================
# 8. Model functions
# ============================================================

def create_model():

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=NUM_CLASSES,
    )

    return model.to(DEVICE)


def collect_logits(
    model,
    data_loader,
):

    model.eval()

    all_logits = []
    all_labels = []
    all_paths = []

    with torch.no_grad():

        for images, labels, attributes, paths in data_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(
                images
            )

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            all_paths.extend(
                list(paths)
            )

    return (
        torch.cat(
            all_logits,
            dim=0,
        ),
        torch.cat(
            all_labels,
            dim=0,
        ),
        all_paths,
    )


def evaluate_logits(
    logits,
    labels,
    temperature=1.0,
):

    probabilities = torch.softmax(
        logits
        / temperature,
        dim=1,
    ).numpy()

    labels_numpy = labels.numpy()

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = probabilities.max(
        axis=1
    )

    correctness = (
        predictions
        == labels_numpy
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "confidence": confidence,
        "correctness": correctness,
        "accuracy": accuracy_score(
            labels_numpy,
            predictions,
        ),
        "macro_f1": f1_score(
            labels_numpy,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "ece": calculate_ece(
            confidence=confidence,
            correctness=correctness,
            number_of_bins=10,
        ),
    }


# ============================================================
# 9. Train one model
# ============================================================

def train_one_seed(
    seed,
    train_loader,
    validation_loader,
    class_weights,
):

    set_seed(seed)

    model = create_model()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_f1 = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        training_losses = []

        for images, labels, attributes, paths in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            training_losses.append(
                loss.item()
            )

        validation_logits, validation_labels, _ = (
            collect_logits(
                model=model,
                data_loader=validation_loader,
            )
        )

        validation_results = evaluate_logits(
            logits=validation_logits,
            labels=validation_labels,
            temperature=1.0,
        )

        validation_f1 = (
            validation_results["macro_f1"]
        )

        print(
            f"Seed {seed} | "
            f"Epoch {epoch:02d} | "
            f"Train loss "
            f"{np.mean(training_losses):.4f} | "
            f"Validation macro F1 "
            f"{validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:

            best_validation_f1 = validation_f1

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping for seed {seed}."
            )

            break

    if best_state is None:
        raise RuntimeError(
            f"No best checkpoint was saved for seed {seed}."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    return (
        model,
        best_epoch,
        best_validation_f1,
    )


# ============================================================
# 10. Create datasets and loaders
# ============================================================

train_dataset = WBCAttDataset(
    csv_path=TRAIN_CSV,
    split="repeated_seed_train",
    transform=training_transform,
)

validation_dataset = WBCAttDataset(
    csv_path=VALIDATION_CSV,
    split="repeated_seed_validation",
    transform=evaluation_transform,
)

test_dataset = WBCAttDataset(
    csv_path=TEST_CSV,
    split="repeated_seed_test",
    transform=evaluation_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 11. Calculate class weights
# ============================================================

train_counts = (
    train_dataframe["label"]
    .value_counts()
)

total_training_samples = len(
    train_dataframe
)

class_weights_list = []

for class_name in CLASS_NAMES:

    class_count = int(
        train_counts.get(
            class_name,
            1,
        )
    )

    class_weight = (
        total_training_samples
        / (
            NUM_CLASSES
            * class_count
        )
    )

    class_weights_list.append(
        class_weight
    )

class_weights = torch.tensor(
    class_weights_list,
    dtype=torch.float32,
    device=DEVICE,
)

print()
print(
    "Class weights:"
)

for class_name, weight in zip(
    CLASS_NAMES,
    class_weights_list,
):

    print(
        f"{class_name}: {weight:.4f}"
    )


# ============================================================
# 12. Repeated training and evaluation
# ============================================================

summary_rows = []
target_group_rows = []

for seed in SEEDS:

    print()
    print("=" * 70)
    print(
        f"STARTING SEED {seed}"
    )
    print("=" * 70)

    model, best_epoch, best_validation_f1 = (
        train_one_seed(
            seed=seed,
            train_loader=train_loader,
            validation_loader=validation_loader,
            class_weights=class_weights,
        )
    )

    validation_logits, validation_labels, _ = (
        collect_logits(
            model=model,
            data_loader=validation_loader,
        )
    )

    test_logits, test_labels, test_paths = (
        collect_logits(
            model=model,
            data_loader=test_loader,
        )
    )

    temperature = fit_temperature(
        validation_logits=validation_logits,
        validation_labels=validation_labels,
    )

    raw_test_results = evaluate_logits(
        logits=test_logits,
        labels=test_labels,
        temperature=1.0,
    )

    calibrated_test_results = evaluate_logits(
        logits=test_logits,
        labels=test_labels,
        temperature=temperature,
    )

    test_image_names = [
        Path(
            str(path)
        ).name
        for path in test_paths
    ]

    prediction_dataframe = pd.DataFrame({
        "img_name": test_image_names,
        "predicted_index": (
            calibrated_test_results[
                "predictions"
            ]
        ),
        "calibrated_confidence": (
            calibrated_test_results[
                "confidence"
            ]
        ),
        "correct": (
            calibrated_test_results[
                "correctness"
            ]
        ),
    })

    prediction_dataframe["predicted_label"] = (
        prediction_dataframe[
            "predicted_index"
        ]
        .map(
            {
                index: class_name
                for index, class_name
                in enumerate(CLASS_NAMES)
            }
        )
    )

    test_metadata = test_dataframe[
        [
            "img_name",
            "label",
            "nuclear_cytoplasmic_ratio",
        ]
    ].copy()

    test_metadata["img_name"] = (
        test_metadata["img_name"]
        .astype(str)
        .str.strip()
    )

    prediction_dataframe["img_name"] = (
        prediction_dataframe["img_name"]
        .astype(str)
        .str.strip()
    )

    merged_test = test_metadata.merge(
        prediction_dataframe,
        on="img_name",
        how="inner",
        validate="one_to_one",
    )

    merged_test["label"] = (
        merged_test["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    merged_test["nuclear_cytoplasmic_ratio"] = (
        merged_test[
            "nuclear_cytoplasmic_ratio"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_mask = (
        (
            merged_test["label"]
            == "monocyte"
        )
        &
        (
            merged_test[
                "nuclear_cytoplasmic_ratio"
            ]
            == "high"
        )
    )

    target_group = (
        merged_test[
            target_mask
        ]
        .copy()
    )

    target_confidence = (
        target_group[
            "calibrated_confidence"
        ]
        .to_numpy(
            dtype=float
        )
    )

    target_correctness = (
        target_group[
            "correct"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    target_accuracy = (
        target_correctness.mean()
    )

    target_mean_confidence = (
        target_confidence.mean()
    )

    target_gap = (
        target_mean_confidence
        - target_accuracy
    )

    target_ece = calculate_ece(
        confidence=target_confidence,
        correctness=target_correctness,
        number_of_bins=10,
    )

    target_group["seed"] = seed
    target_group["temperature"] = temperature
    target_group["group_ece"] = target_ece

    target_group_rows.append(
        target_group
    )

    summary_rows.append(
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": (
                best_validation_f1
            ),
            "temperature": temperature,

            "raw_test_accuracy": (
                raw_test_results["accuracy"]
            ),
            "raw_test_macro_f1": (
                raw_test_results["macro_f1"]
            ),
            "raw_test_ece": (
                raw_test_results["ece"]
            ),

            "calibrated_test_accuracy": (
                calibrated_test_results[
                    "accuracy"
                ]
            ),
            "calibrated_test_macro_f1": (
                calibrated_test_results[
                    "macro_f1"
                ]
            ),
            "calibrated_test_ece": (
                calibrated_test_results[
                    "ece"
                ]
            ),

            "target_group_sample_count": (
                len(target_group)
            ),
            "target_group_accuracy": (
                target_accuracy
            ),
            "target_group_mean_confidence": (
                target_mean_confidence
            ),
            "target_group_confidence_accuracy_gap": (
                target_gap
            ),
            "target_group_ece": (
                target_ece
            ),
            "target_group_incorrect_count": int(
                (~target_correctness).sum()
            ),
        }
    )

    print()
    print(
        f"Seed {seed} completed."
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Temperature: {temperature:.6f}"
    )

    print(
        "Calibrated test accuracy: "
        f"{calibrated_test_results['accuracy']:.4f}"
    )

    print(
        "Calibrated test ECE: "
        f"{calibrated_test_results['ece']:.4f}"
    )

    print(
        "Target-group accuracy: "
        f"{target_accuracy:.4f}"
    )

    print(
        "Target-group ECE: "
        f"{target_ece:.4f}"
    )


# ============================================================
# 13. Save results
# ============================================================

summary_dataframe = pd.DataFrame(
    summary_rows
)

summary_dataframe.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

target_group_dataframe = pd.concat(
    target_group_rows,
    ignore_index=True,
)

target_group_dataframe.to_csv(
    TARGET_GROUP_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. Create summary plot
# ============================================================

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5),
)

axes[0].bar(
    summary_dataframe["seed"].astype(str),
    summary_dataframe[
        "target_group_accuracy"
    ],
    color="steelblue",
    edgecolor="black",
)

axes[0].set_ylim(
    0,
    1.05,
)

axes[0].set_title(
    "Target-Group Accuracy by Seed"
)

axes[0].set_xlabel(
    "Training seed"
)

axes[0].set_ylabel(
    "Accuracy"
)

axes[1].bar(
    summary_dataframe["seed"].astype(str),
    summary_dataframe[
        "target_group_ece"
    ],
    color="darkorange",
    edgecolor="black",
)

axes[1].set_title(
    "Target-Group ECE by Seed"
)

axes[1].set_xlabel(
    "Training seed"
)

axes[1].set_ylabel(
    "ECE"
)

figure.suptitle(
    "Repeated-Seed Validation of the Monocyte High-NCR Group",
    fontsize=14,
)

figure.tight_layout(
    rect=[
        0,
        0,
        1,
        0.92,
    ]
)

figure.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# 15. Print final summary
# ============================================================

print()
print("=" * 70)
print("REPEATED-SEED VALIDATION FINISHED")
print("=" * 70)

print()
print(
    summary_dataframe[
        [
            "seed",
            "calibrated_test_accuracy",
            "calibrated_test_ece",
            "target_group_accuracy",
            "target_group_ece",
            "target_group_incorrect_count",
        ]
    ].to_string(index=False)
)

print()
print(
    "Mean target-group accuracy: "
    f"{summary_dataframe['target_group_accuracy'].mean():.4f}"
)

print(
    "Standard deviation of target-group accuracy: "
    f"{summary_dataframe['target_group_accuracy'].std(ddof=1):.4f}"
)

print(
    "Mean target-group ECE: "
    f"{summary_dataframe['target_group_ece'].mean():.4f}"
)

print(
    "Standard deviation of target-group ECE: "
    f"{summary_dataframe['target_group_ece'].std(ddof=1):.4f}"
)

print()
print(
    f"Summary saved to:\n{SUMMARY_PATH}"
)

print()
print(
    f"Target-group predictions saved to:\n"
    f"{TARGET_GROUP_PATH}"
)

print()
print(
    f"Plot saved to:\n{PLOT_PATH}"
)
