
from pathlib import Path
import math
import sys

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


sys.path.insert(
    0,
    "/content"
)

from mafd_utils import (
    ALL_CASES_PATH,
    FAILURE_ANALYSIS_DIR,
    clean_prediction_dataframe,
    safe_filename,
)


# ============================================================
# Output paths
# ============================================================

COMPARISON_PATH = (
    FAILURE_ANALYSIS_DIR
    / "failure_control_comparison.csv"
)

COMPARISON_SHEET_PATH = (
    FAILURE_ANALYSIS_DIR
    / "failure_control_comparison_sheet.png"
)


# ============================================================
# Load cases
# ============================================================

if not ALL_CASES_PATH.exists():
    raise FileNotFoundError(
        f"Required file was not found:\n{ALL_CASES_PATH}"
    )

all_cases = pd.read_csv(
    ALL_CASES_PATH
)

all_cases = clean_prediction_dataframe(
    all_cases
)

required_columns = [
    "img_name",
    "resolved_path",
    "true_label",
    "predicted_label",
    "calibrated_confidence",
    "correct",
]

missing_columns = [
    column
    for column in required_columns
    if column not in all_cases.columns
]

if missing_columns:
    raise KeyError(
        "The risk-group CSV is missing:\n"
        f"{missing_columns}"
    )


# ============================================================
# Separate failures and correct controls
# ============================================================

failure_cases = all_cases[
    all_cases["correct"] == False
].copy()

correct_cases = all_cases[
    all_cases["correct"] == True
].copy()

if len(failure_cases) == 0:
    raise RuntimeError(
        "No failure cases were found."
    )

if len(correct_cases) == 0:
    raise RuntimeError(
        "No correct control cases were found."
    )


# Select five reproducible controls.
number_of_controls = min(
    5,
    len(correct_cases)
)

correct_controls = correct_cases.sample(
    n=number_of_controls,
    random_state=42
).copy()

failure_cases["case_type"] = (
    "confirmed_failure"
)

correct_controls["case_type"] = (
    "correct_control"
)

failure_cases["comparison_order"] = 0
correct_controls["comparison_order"] = 1

comparison_dataframe = pd.concat(
    [
        failure_cases,
        correct_controls,
    ],
    ignore_index=True,
)

comparison_dataframe = (
    comparison_dataframe
    .sort_values(
        by=[
            "comparison_order",
            "calibrated_confidence",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .drop(
        columns=[
            "comparison_order",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# Add manual comparison columns
# ============================================================

manual_columns = [
    "same_visual_pattern_as_controls",
    "nucleus_difference",
    "cytoplasm_difference",
    "size_difference",
    "shape_difference",
    "background_difference",
    "manual_comparison_notes",
]

for column in manual_columns:
    comparison_dataframe[column] = ""


comparison_dataframe.to_csv(
    COMPARISON_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Create comparison contact sheet
# ============================================================

number_of_images = len(
    comparison_dataframe
)

number_of_columns = min(
    3,
    number_of_images
)

number_of_rows = math.ceil(
    number_of_images
    / number_of_columns
)

figure, axes = plt.subplots(
    nrows=number_of_rows,
    ncols=number_of_columns,
    figsize=(
        5 * number_of_columns,
        6 * number_of_rows,
    ),
)

if hasattr(axes, "flatten"):
    axes = axes.flatten()
else:
    axes = [axes]

for image_index, (_, row) in enumerate(
    comparison_dataframe.iterrows()
):

    axis = axes[image_index]

    image_path = Path(
        row["resolved_path"]
    )

    try:
        image = Image.open(
            image_path
        ).convert("RGB")

        image = ImageOps.contain(
            image,
            size=(800, 800)
        )

        axis.imshow(image)

    except Exception as error:

        axis.text(
            0.5,
            0.5,
            f"Could not open image:\n{error}",
            ha="center",
            va="center",
        )

    axis.axis("off")

    case_type = row["case_type"]
    color = "red" if case_type == (
        "confirmed_failure"
    ) else "green"

    title = (
        f"{case_type}\n"
        f"{row['img_name']}\n"
        f"True: {row['true_label']} | "
        f"Predicted: {row['predicted_label']}\n"
        f"Confidence: "
        f"{float(row['calibrated_confidence']):.4f}"
    )

    axis.set_title(
        title,
        color=color,
        fontsize=10,
    )

for unused_axis in axes[number_of_images:]:
    unused_axis.axis("off")

figure.suptitle(
    "Confirmed Failures and Correct Controls\n"
    "Monocyte with High Nuclear-Cytoplasmic Ratio",
    fontsize=15,
)

figure.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)

figure.savefig(
    COMPARISON_SHEET_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# Print summary
# ============================================================

print("=" * 70)
print("FAILURE AND CONTROL COMPARISON CREATED")
print("=" * 70)

print()
print(
    f"Confirmed failures: {len(failure_cases)}"
)

print(
    f"Correct controls: {len(correct_controls)}"
)

print()
print(
    f"Comparison CSV:\n{COMPARISON_PATH}"
)

print()
print(
    f"Comparison contact sheet:\n"
    f"{COMPARISON_SHEET_PATH}"
)

print()
print(
    "Inspect whether the failures look different from "
    "the correctly classified monocytes."
)
