from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

DATASET_ROOT = Path(
    "/content/WBCAtt"
)

MODEL_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/MAFD_OUTPUTS/model_a_clean"
)

if not MODEL_OUTPUT_DIR.exists():
    MODEL_OUTPUT_DIR = Path(
        "/content/MAFD_OUTPUTS/model_a_clean"
    )


CLEAN_TEST_CSV = (
    MODEL_OUTPUT_DIR
    / "clean_annotations"
    / "test_clean.csv"
)

PREDICTIONS_CSV = (
    MODEL_OUTPUT_DIR
    / "model_a_clean_test_predictions.csv"
)

ANALYSIS_OUTPUT_DIR = (
    MODEL_OUTPUT_DIR
    / "morphology_reliability_analysis"
)

ANALYSIS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GROUP_RESULTS_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "attribute_value_reliability.csv"
)

ATTRIBUTE_SUMMARY_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "attribute_reliability_summary.csv"
)


# ============================================================
# 2. Dataset definitions
# ============================================================

LABEL_COLUMN = "label"
IMAGE_COLUMN = "path"

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


# ============================================================
# 3. Check required files
# ============================================================

if not CLEAN_TEST_CSV.exists():
    raise FileNotFoundError(
        f"Clean test CSV was not found:\n"
        f"{CLEAN_TEST_CSV}"
    )

if not PREDICTIONS_CSV.exists():
    raise FileNotFoundError(
        f"Prediction CSV was not found:\n"
        f"{PREDICTIONS_CSV}"
    )


# ============================================================
# 4. Resolve image paths
# ============================================================

def resolve_image_path(raw_value, label):
    """
    Resolves an image path from the cleaned CSV.
    """

    value = str(raw_value).strip()
    value = value.replace("\\", "/")

    raw_path = Path(value)

    image_root = (
        DATASET_ROOT
        / "PBC_dataset_normal_DIB"
    )

    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)

    candidates.extend(
        [
            raw_path,
            DATASET_ROOT / raw_path,
            image_root / raw_path,
            image_root / label / raw_path,
            image_root / label / raw_path.name,
            image_root / raw_path.name,
            DATASET_ROOT / raw_path.name,
        ]
    )

    checked = set()

    for candidate in candidates:

        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        candidate_key = str(candidate).lower()

        if candidate_key in checked:
            continue

        checked.add(candidate_key)

        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not resolve image path.\n"
        f"CSV value: {raw_value}\n"
        f"Label: {label}"
    )


def normalize_path(path):
    """
    Creates a consistent path key for merging.
    """

    return str(
        Path(path).resolve()
    ).replace("\\", "/").lower()


# ============================================================
# 5. Expected calibration error
# ============================================================

def calculate_ece(
    confidence_values,
    correctness_values,
    number_of_bins=10,
):
    """
    Calculates expected calibration error.

    A well-calibrated model should have similar values for:
        average confidence
        actual accuracy
    """

    confidence_values = np.asarray(
        confidence_values,
        dtype=float,
    )

    correctness_values = np.asarray(
        correctness_values,
        dtype=float,
    )

    if len(confidence_values) == 0:
        return np.nan

    ece = 0.0

    for bin_index in range(number_of_bins):

        lower_bound = (
            bin_index
            / number_of_bins
        )

        upper_bound = (
            bin_index + 1
        ) / number_of_bins

        if bin_index == number_of_bins - 1:

            in_bin = (
                (confidence_values >= lower_bound)
                & (confidence_values <= upper_bound)
            )

        else:

            in_bin = (
                (confidence_values >= lower_bound)
                & (confidence_values < upper_bound)
            )

        if not np.any(in_bin):
            continue

        bin_confidence = (
            confidence_values[in_bin].mean()
        )

        bin_accuracy = (
            correctness_values[in_bin].mean()
        )

        bin_fraction = (
            in_bin.sum()
            / len(confidence_values)
        )

        ece += (
            bin_fraction
            * abs(bin_confidence - bin_accuracy)
        )

    return float(ece)


# ============================================================
# 6. Load data
# ============================================================

print("=" * 70)
print("MORPHOLOGY-SPECIFIC CONFIDENCE RELIABILITY ANALYSIS")
print("=" * 70)

print()
print(
    f"Clean test CSV:\n{CLEAN_TEST_CSV}"
)

print()
print(
    f"Prediction CSV:\n{PREDICTIONS_CSV}"
)

test_dataframe = pd.read_csv(
    CLEAN_TEST_CSV
)

prediction_dataframe = pd.read_csv(
    PREDICTIONS_CSV
)

print()
print(
    f"Clean test rows: "
    f"{len(test_dataframe)}"
)

print(
    f"Prediction rows: "
    f"{len(prediction_dataframe)}"
)


# ============================================================
# 7. Prepare prediction columns
# ============================================================

required_prediction_columns = [
    "image_path",
    "confidence",
    "correct",
]

missing_prediction_columns = [
    column
    for column in required_prediction_columns
    if column not in prediction_dataframe.columns
]

if missing_prediction_columns:
    raise ValueError(
        "The prediction file is missing columns:\n"
        f"{missing_prediction_columns}"
    )


prediction_dataframe["confidence"] = pd.to_numeric(
    prediction_dataframe["confidence"],
    errors="coerce",
)

prediction_dataframe["correct"] = (
    prediction_dataframe["correct"]
    .astype(str)
    .str.lower()
    .eq("true")
)

prediction_dataframe["normalized_path"] = (
    prediction_dataframe["image_path"]
    .apply(normalize_path)
)


# ============================================================
# 8. Resolve test paths and merge predictions
# ============================================================

test_dataframe["resolved_path"] = (
    test_dataframe.apply(
        lambda row: str(
            resolve_image_path(
                raw_value=row[IMAGE_COLUMN],
                label=str(
                    row[LABEL_COLUMN]
                ).strip().lower(),
            )
        ),
        axis=1,
    )
)

test_dataframe["normalized_path"] = (
    test_dataframe["resolved_path"]
    .apply(normalize_path)
)

merged_dataframe = test_dataframe.merge(
    prediction_dataframe[
        [
            "normalized_path",
            "confidence",
            "correct",
            "predicted_label",
        ]
    ],
    on="normalized_path",
    how="left",
)


missing_predictions = (
    merged_dataframe["confidence"]
    .isna()
    .sum()
)

if missing_predictions > 0:

    print()
    print(
        "Warning: predictions could not be matched "
        f"for {missing_predictions} test images."
    )

    merged_dataframe = merged_dataframe.dropna(
        subset=[
            "confidence",
            "correct",
        ]
    ).copy()


if len(merged_dataframe) == 0:
    raise RuntimeError(
        "No test predictions could be matched "
        "with the clean test CSV."
    )


print()
print(
    f"Matched test predictions: "
    f"{len(merged_dataframe)}"
)


# ============================================================
# 9. Overall calibration result
# ============================================================

overall_accuracy = (
    merged_dataframe["correct"]
    .mean()
)

overall_mean_confidence = (
    merged_dataframe["confidence"]
    .mean()
)

overall_ece = calculate_ece(
    confidence_values=merged_dataframe[
        "confidence"
    ].values,

    correctness_values=merged_dataframe[
        "correct"
    ].astype(int).values,
)

print()
print("=" * 70)
print("OVERALL CONFIDENCE RESULTS")
print("=" * 70)

print(
    f"Accuracy:              "
    f"{overall_accuracy:.4f}"
)

print(
    f"Mean confidence:       "
    f"{overall_mean_confidence:.4f}"
)

print(
    f"Expected calibration error: "
    f"{overall_ece:.4f}"
)


# ============================================================
# 10. Analyze every morphological attribute value
# ============================================================

group_results = []

for attribute_name in ATTRIBUTE_COLUMNS:

    attribute_groups = merged_dataframe.groupby(
        attribute_name,
        dropna=False,
    )

    for attribute_value, group in attribute_groups:

        group_size = len(group)

        group_accuracy = (
            group["correct"]
            .mean()
        )

        group_confidence = (
            group["confidence"]
            .mean()
        )

        group_ece = calculate_ece(
            confidence_values=group[
                "confidence"
            ].values,

            correctness_values=group[
                "correct"
            ].astype(int).values,
        )

        correct_rows = group[
            group["correct"] == True
        ]

        incorrect_rows = group[
            group["correct"] == False
        ]

        if len(correct_rows) > 0:
            confidence_when_correct = (
                correct_rows["confidence"]
                .mean()
            )
        else:
            confidence_when_correct = np.nan

        if len(incorrect_rows) > 0:
            confidence_when_incorrect = (
                incorrect_rows["confidence"]
                .mean()
            )
        else:
            confidence_when_incorrect = np.nan

        confidence_accuracy_gap = (
            group_confidence
            - group_accuracy
        )

        group_results.append(
            {
                "attribute": attribute_name,
                "attribute_value": str(
                    attribute_value
                ),
                "sample_count": group_size,
                "accuracy": group_accuracy,
                "error_rate": 1.0 - group_accuracy,
                "mean_confidence": group_confidence,
                "confidence_when_correct": (
                    confidence_when_correct
                ),
                "confidence_when_incorrect": (
                    confidence_when_incorrect
                ),
                "confidence_accuracy_gap": (
                    confidence_accuracy_gap
                ),
                "expected_calibration_error": (
                    group_ece
                ),
            }
        )


group_results_dataframe = pd.DataFrame(
    group_results
)

group_results_dataframe = (
    group_results_dataframe.sort_values(
        by=[
            "expected_calibration_error",
            "confidence_accuracy_gap",
        ],
        ascending=[
            False,
            False,
        ],
    )
)

group_results_dataframe.to_csv(
    GROUP_RESULTS_PATH,
    index=False,
)


# ============================================================
# 11. Attribute-level summary
# ============================================================

attribute_summaries = []

for attribute_name in ATTRIBUTE_COLUMNS:

    attribute_data = group_results_dataframe[
        group_results_dataframe["attribute"]
        == attribute_name
    ].copy()

    weighted_accuracy = np.average(
        attribute_data["accuracy"],
        weights=attribute_data["sample_count"],
    )

    weighted_confidence = np.average(
        attribute_data["mean_confidence"],
        weights=attribute_data["sample_count"],
    )

    weighted_ece = np.average(
        attribute_data[
            "expected_calibration_error"
        ].fillna(0),
        weights=attribute_data["sample_count"],
    )

    highest_confidence_gap_row = (
        attribute_data.iloc[
            attribute_data[
                "confidence_accuracy_gap"
            ].abs().argmax()
        ]
    )

    worst_ece_row = (
        attribute_data.iloc[
            attribute_data[
                "expected_calibration_error"
            ].argmax()
        ]
    )

    attribute_summaries.append(
        {
            "attribute": attribute_name,
            "number_of_values": len(
                attribute_data
            ),
            "total_samples": int(
                attribute_data["sample_count"]
                .sum()
            ),
            "weighted_accuracy": (
                weighted_accuracy
            ),
            "weighted_mean_confidence": (
                weighted_confidence
            ),
            "weighted_expected_calibration_error": (
                weighted_ece
            ),
            "largest_confidence_accuracy_gap_value": (
                highest_confidence_gap_row[
                    "attribute_value"
                ]
            ),
            "largest_confidence_accuracy_gap": (
                highest_confidence_gap_row[
                    "confidence_accuracy_gap"
                ]
            ),
            "highest_ece_value": (
                worst_ece_row[
                    "attribute_value"
                ]
            ),
            "highest_ece": (
                worst_ece_row[
                    "expected_calibration_error"
                ]
            ),
        }
    )


attribute_summary_dataframe = pd.DataFrame(
    attribute_summaries
)

attribute_summary_dataframe = (
    attribute_summary_dataframe.sort_values(
        by="weighted_expected_calibration_error",
        ascending=False,
    )
)

attribute_summary_dataframe.to_csv(
    ATTRIBUTE_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 12. Print results
# ============================================================

print()
print("=" * 70)
print("ATTRIBUTE-LEVEL SUMMARY")
print("=" * 70)

print(
    attribute_summary_dataframe[
        [
            "attribute",
            "weighted_accuracy",
            "weighted_mean_confidence",
            "weighted_expected_calibration_error",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 70)
print("WORST ATTRIBUTE VALUES BY CALIBRATION ERROR")
print("=" * 70)

print(
    group_results_dataframe[
        [
            "attribute",
            "attribute_value",
            "sample_count",
            "accuracy",
            "mean_confidence",
            "confidence_accuracy_gap",
            "expected_calibration_error",
        ]
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# 13. Create plots for every attribute
# ============================================================

for attribute_name in ATTRIBUTE_COLUMNS:

    attribute_data = group_results_dataframe[
        group_results_dataframe["attribute"]
        == attribute_name
    ].copy()

    attribute_data = attribute_data.sort_values(
        by="attribute_value"
    )

    values = attribute_data[
        "attribute_value"
    ].tolist()

    accuracy_values = attribute_data[
        "accuracy"
    ].tolist()

    confidence_values = attribute_data[
        "mean_confidence"
    ].tolist()

    ece_values = attribute_data[
        "expected_calibration_error"
    ].tolist()

    x_positions = np.arange(
        len(values)
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )

    axes[0].bar(
        x_positions - 0.2,
        accuracy_values,
        width=0.4,
        label="Accuracy",
    )

    axes[0].bar(
        x_positions + 0.2,
        confidence_values,
        width=0.4,
        label="Mean confidence",
    )

    axes[0].set_ylim(
        0,
        1.05,
    )

    axes[0].set_xticks(
        x_positions
    )

    axes[0].set_xticklabels(
        values,
        rotation=45,
        ha="right",
    )

    axes[0].set_ylabel(
        "Value"
    )

    axes[0].set_title(
        f"{attribute_name}: Accuracy vs Confidence"
    )

    axes[0].legend()

    axes[1].bar(
        x_positions,
        ece_values,
        color="darkorange",
    )

    axes[1].set_xticks(
        x_positions
    )

    axes[1].set_xticklabels(
        values,
        rotation=45,
        ha="right",
    )

    axes[1].set_ylabel(
        "Expected calibration error"
    )

    axes[1].set_title(
        f"{attribute_name}: Calibration Error"
    )

    figure.tight_layout()

    safe_name = (
        attribute_name
        .replace(" ", "_")
    )

    plot_path = (
        ANALYSIS_OUTPUT_DIR
        / f"{safe_name}_reliability.png"
    )

    figure.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 14. Final output
# ============================================================

print()
print("=" * 70)
print("MORPHOLOGY RELIABILITY ANALYSIS FINISHED")
print("=" * 70)

print()
print(
    f"Group-level results saved to:\n"
    f"{GROUP_RESULTS_PATH}"
)

print()
print(
    f"Attribute-level summary saved to:\n"
    f"{ATTRIBUTE_SUMMARY_PATH}"
)

print()
print(
    f"Plots saved to:\n"
    f"{ANALYSIS_OUTPUT_DIR}"
)
