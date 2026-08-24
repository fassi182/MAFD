from pathlib import Path
import importlib.util
import random
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
# 2. Paths and training configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/MAFD_OUTPUTS"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BEST_MODEL_PATH = OUTPUT_DIR / "model_a_resnet18_best.pt"
HISTORY_PATH = OUTPUT_DIR / "model_a_training_history.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "model_a_test_predictions.csv"
TEST_REPORT_PATH = OUTPUT_DIR / "model_a_test_classification_report.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "model_a_test_confusion_matrix.png"
CONFIG_PATH = OUTPUT_DIR / "model_a_config.json"


BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 5

IMAGE_SIZE = 224
NUM_CLASSES = 5

USE_PRETRAINED_WEIGHTS = True


# ============================================================
# 3. Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("MODEL A: WBC FIVE-CLASS CLASSIFIER")
print("=" * 70)
print()
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# 4. Import the Dataset from the previous file
# ============================================================

def load_dataset_module():
    """
    Loads the Dataset and DataLoader definitions from the
    previous script.

    The code supports either filename:
        2_test_dataloader.py
        02_test_dataloader.py
    """

    possible_files = [
        PROJECT_ROOT / "2_test_dataloader.py",
        PROJECT_ROOT / "02_test_dataloader.py",
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
            f"  {possible_files[0]}\n"
            f"  {possible_files[1]}"
        )

    module_name = "wbc_dataset_module"

    specification = importlib.util.spec_from_file_location(
        module_name,
        dataset_file,
    )

    if specification is None:
        raise ImportError(
            f"Could not create import specification for:\n"
            f"{dataset_file}"
        )

    dataset_module = importlib.util.module_from_spec(
        specification
    )

    if specification.loader is None:
        raise ImportError(
            f"Could not load dataset module:\n"
            f"{dataset_file}"
        )

    specification.loader.exec_module(
        dataset_module
    )

    return dataset_module


dataset_module = load_dataset_module()

WBCAttDataset = dataset_module.WBCAttDataset
CSV_PATHS = dataset_module.CSV_PATHS
CLASS_NAMES = dataset_module.CLASS_NAMES
train_transform = dataset_module.train_transform
evaluation_transform = dataset_module.evaluation_transform


# ============================================================
# 5. Create datasets and DataLoaders
# ============================================================

print()
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dataset = WBCAttDataset(
    csv_path=CSV_PATHS["train"],
    split="train",
    transform=train_transform,
)

validation_dataset = WBCAttDataset(
    csv_path=CSV_PATHS["validation"],
    split="validation",
    transform=evaluation_transform,
)

test_dataset = WBCAttDataset(
    csv_path=CSV_PATHS["test"],
    split="test",
    transform=evaluation_transform,
)


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


print()
print("Dataset sizes:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test: {len(test_dataset)}")


# ============================================================
# 6. Calculate class weights
# ============================================================

def calculate_class_weights(dataset):
    """
    Calculates inverse-frequency class weights.

    This helps reduce the effect of class imbalance during
    training. The test set remains untouched.
    """

    labels = dataset.labels

    class_counts = torch.bincount(
        labels,
        minlength=NUM_CLASSES,
    ).float()

    total_samples = class_counts.sum()

    class_weights = total_samples / (
        NUM_CLASSES * class_counts
    )

    return class_counts, class_weights


train_class_counts, class_weights = calculate_class_weights(
    train_dataset
)

print()
print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

for index, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:15s} "
        f"count={int(train_class_counts[index].item()):5d} "
        f"weight={class_weights[index].item():.4f}"
    )


# ============================================================
# 7. Build Model A
# ============================================================

def build_model():
    """
    Creates a ResNet18 model with five output classes.
    """

    if USE_PRETRAINED_WEIGHTS:

        try:
            print()
            print("Loading pretrained ResNet18 weights...")

            model = resnet18(
                weights=ResNet18_Weights.DEFAULT
            )

            print(
                "Pretrained weights loaded successfully."
            )

        except Exception as error:

            print()
            print(
                "Pretrained weights could not be loaded."
            )
            print(
                "The model will be initialized from scratch."
            )
            print(f"Reason: {error}")

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
        out_features=NUM_CLASSES,
    )

    return model


model = build_model()
model = model.to(DEVICE)

print()
print("=" * 70)
print("MODEL INFORMATION")
print("=" * 70)
print(model.fc)


# ============================================================
# 8. Loss function, optimizer, and scheduler
# ============================================================

class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = AdamW(
    params=model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# 9. Metric functions
# ============================================================

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculates classification metrics.
    """

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# ============================================================
# 10. Train and evaluation functions
# ============================================================

def run_one_epoch(
    model,
    data_loader,
    criterion,
    device,
    optimizer=None,
):
    """
    Runs one training or evaluation epoch.

    If optimizer is provided, training is performed.
    If optimizer is None, evaluation is performed.
    """

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
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):

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
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_probabilities.append(
            probabilities.detach().cpu()
        )

    average_loss = total_loss / total_samples

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

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
# 11. Training loop
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = []

best_validation_loss = float("inf")
best_validation_macro_f1 = -1.0
epochs_without_improvement = 0

for epoch in range(1, NUM_EPOCHS + 1):

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

    current_learning_rate = optimizer.param_groups[0]["lr"]

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

    epoch_record = {
        "epoch": epoch,
        "learning_rate": current_learning_rate,

        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": train_metrics[
            "balanced_accuracy"
        ],
        "train_macro_f1": train_metrics["macro_f1"],
        "train_weighted_f1": train_metrics["weighted_f1"],
        "train_macro_precision": train_metrics[
            "macro_precision"
        ],
        "train_macro_recall": train_metrics[
            "macro_recall"
        ],

        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics[
            "accuracy"
        ],
        "validation_balanced_accuracy": validation_metrics[
            "balanced_accuracy"
        ],
        "validation_macro_f1": validation_metrics[
            "macro_f1"
        ],
        "validation_weighted_f1": validation_metrics[
            "weighted_f1"
        ],
        "validation_macro_precision": validation_metrics[
            "macro_precision"
        ],
        "validation_macro_recall": validation_metrics[
            "macro_recall"
        ],
    }

    history.append(epoch_record)

    # Save the model when validation macro F1 improves.
    if validation_metrics["macro_f1"] > best_validation_macro_f1:

        best_validation_macro_f1 = validation_metrics[
            "macro_f1"
        ]

        best_validation_loss = validation_metrics[
            "loss"
        ]

        epochs_without_improvement = 0

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "best_validation_macro_f1": best_validation_macro_f1,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "seed": SEED,
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH,
        )

        print()
        print(
            "Best model checkpoint saved."
        )

    else:
        epochs_without_improvement += 1

        print()
        print(
            f"No validation macro F1 improvement. "
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
# 12. Save training history
# ============================================================

history_dataframe = pd.DataFrame(history)

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
# 13. Load the best validation checkpoint
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print(
    f"Best checkpoint epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation macro F1: "
    f"{checkpoint['best_validation_macro_f1']:.4f}"
)


# ============================================================
# 14. Final test evaluation
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

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
# 15. Per-class classification report
# ============================================================

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
    TEST_REPORT_PATH
)

print()
print("=" * 70)
print("PER-CLASS TEST RESULTS")
print("=" * 70)
print()
print(
    classification_report(
        y_true=test_labels,
        y_pred=test_predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)


# ============================================================
# 16. Save test predictions and confidence
# ============================================================

test_image_paths = [
    str(path)
    for path in test_dataset.image_paths
]

maximum_confidence = test_probabilities.max(
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
        "predicted_label_index": test_predictions,
        "predicted_label": [
            CLASS_NAMES[index]
            for index in test_predictions
        ],
        "confidence": maximum_confidence,
        "correct": (
            test_labels == test_predictions
        ),
    }
)

for class_index, class_name in enumerate(CLASS_NAMES):

    probability_column = (
        f"probability_{class_name}"
    )

    prediction_dataframe[probability_column] = (
        test_probabilities[:, class_index]
    )

prediction_dataframe.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False,
)

print(
    f"Test predictions saved to:\n"
    f"{TEST_PREDICTIONS_PATH}"
)


# ============================================================
# 17. Create confusion matrix
# ============================================================

confusion = confusion_matrix(
    y_true=test_labels,
    y_pred=test_predictions,
    labels=list(range(NUM_CLASSES)),
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
    "Model A Test Confusion Matrix"
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
# 18. Save configuration and summary
# ============================================================

configuration = {
    "seed": SEED,
    "device": str(DEVICE),
    "batch_size": BATCH_SIZE,
    "num_epochs_requested": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "architecture": "ResNet18",
    "pretrained_weights_requested": USE_PRETRAINED_WEIGHTS,
    "train_samples": len(train_dataset),
    "validation_samples": len(validation_dataset),
    "test_samples": len(test_dataset),
    "best_checkpoint_epoch": int(
        checkpoint["epoch"]
    ),
    "best_validation_macro_f1": float(
        checkpoint["best_validation_macro_f1"]
    ),
    "test_loss": float(
        test_metrics["loss"]
    ),
    "test_accuracy": float(
        test_metrics["accuracy"]
    ),
    "test_balanced_accuracy": float(
        test_metrics["balanced_accuracy"]
    ),
    "test_macro_f1": float(
        test_metrics["macro_f1"]
    ),
    "test_weighted_f1": float(
        test_metrics["weighted_f1"]
    ),
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
# 19. Final output
# ============================================================

print()
print("=" * 70)
print("MODEL A TRAINING FINISHED SUCCESSFULLY")
print("=" * 70)

print()
print("Generated files:")

print(
    f"Best model:\n"
    f"{BEST_MODEL_PATH}"
)

print(
    f"Training history:\n"
    f"{HISTORY_PATH}"
)

print(
    f"Test predictions:\n"
    f"{TEST_PREDICTIONS_PATH}"
)

print(
    f"Classification report:\n"
    f"{TEST_REPORT_PATH}"
)

print(
    f"Confusion matrix:\n"
    f"{CONFUSION_MATRIX_PATH}"
)

print(
    f"Configuration:\n"
    f"{CONFIG_PATH}"
)
