
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision.models import resnet18
from torchvision import transforms


# ============================================================
# Paths
# ============================================================

PRIMARY_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/MAFD_OUTPUTS/model_a_clean"
)

FALLBACK_OUTPUT_DIR = Path(
    "/content/MAFD_OUTPUTS/model_a_clean"
)

if PRIMARY_OUTPUT_DIR.exists():
    MODEL_OUTPUT_DIR = PRIMARY_OUTPUT_DIR
else:
    MODEL_OUTPUT_DIR = FALLBACK_OUTPUT_DIR

if not MODEL_OUTPUT_DIR.exists():
    raise FileNotFoundError(
        f"Model output directory was not found:\n"
        f"{MODEL_OUTPUT_DIR}"
    )


MODEL_PATH = (
    MODEL_OUTPUT_DIR
    / "model_a_clean_resnet18_best.pt"
)

FAILURE_ANALYSIS_DIR = (
    MODEL_OUTPUT_DIR
    / "failure_case_analysis"
)

ALL_CASES_PATH = (
    FAILURE_ANALYSIS_DIR
    / "monocyte_high_ncr_all_cases.csv"
)

FAILURE_CASES_PATH = (
    FAILURE_ANALYSIS_DIR
    / "monocyte_high_ncr_failure_cases.csv"
)

CALIBRATION_DIR = (
    MODEL_OUTPUT_DIR
    / "calibrated_reliability_analysis"
)

CALIBRATION_SUMMARY_PATH = (
    CALIBRATION_DIR
    / "calibration_summary.csv"
)

ATTRIBUTE_RESULTS_PATH = (
    CALIBRATION_DIR
    / "calibrated_attribute_value_reliability.csv"
)


# ============================================================
# Model configuration
# ============================================================

CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_evaluation_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
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
    ])


def load_model():
    """
    Loads the trained Model A ResNet-18 checkpoint.
    """

    device = get_device()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint was not found:\n{MODEL_PATH}"
        )

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=len(CLASS_NAMES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict
    )

    model = model.to(device)
    model.eval()

    return model, device


# ============================================================
# Data utilities
# ============================================================

def parse_boolean(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
        "y",
        "correct",
    }


def clean_prediction_dataframe(dataframe):
    dataframe = dataframe.copy()

    if "correct" in dataframe.columns:
        dataframe["correct"] = (
            dataframe["correct"]
            .apply(parse_boolean)
        )

    if "calibrated_confidence" in dataframe.columns:
        dataframe["calibrated_confidence"] = (
            pd.to_numeric(
                dataframe["calibrated_confidence"],
                errors="coerce",
            )
        )

    if "true_label" in dataframe.columns:
        dataframe["true_label"] = (
            dataframe["true_label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "predicted_label" in dataframe.columns:
        dataframe["predicted_label"] = (
            dataframe["predicted_label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return dataframe


def safe_filename(value):
    value = str(value)

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value


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
