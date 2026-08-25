from pathlib import Path
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

CALIBRATED_PREDICTIONS_CSV = (
    MODEL_OUTPUT_DIR
    / "calibrated_reliability_analysis"
    / "calibrated_test_predictions.csv"
)

OUTPUT_DIR = (
    MODEL_OUTPUT_DIR
    / "class_controlled_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


GROUP_RESULTS_PATH = (
    OUTPUT_DIR
    / "class_controlled_morphology_reliability.csv"
)

CLASS_RESULTS_PATH = (
    OUTPUT_DIR
    / "class_baseline_reliability.csv"
)

TOP_RISK_GROUPS_PATH = (
    OUTPUT_DIR
    / "top_class_controlled_risk_groups.csv"
)

ECE_PLOT_PATH = (
    OUTPUT_DIR
    / "class_controlled_ece.png"
)

ACCURACY_DIFF_PLOT_PATH = (
    OUTPUT_DIR
    / "class_controlled_accuracy_difference.png"
)


# ============================================================
# 2. Dataset definitions
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


# ============================================================
# 3. Check files
# ============================================================

required_files = [
    CLEAN_TEST_CSV,
    CALIBRATED_PREDICTIONS_CSV,
]

for required_file in required_files:

    if not required_file.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n"
            f"{required_file}"
        )


# ============================================================
# 4. Image-path resolution
# ============================================================

def resolve_image_path(raw_value, label):
    """
    Resolves an image path from the clean test CSV.
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
    Calculates group-level expected calibration error.
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
            * abs(
                bin_confidence
                - bin_accuracy
            )
        )

    return float(ece)


# ============================================================
# 6. Load and merge data
# ============================================================

print("=" * 70)
print("CLASS-CONTROLLED MORPHOLOGY RELIABILITY ANALYSIS")
print("=" * 70)

test_dataframe = pd.read_csv(
    CLEAN_TEST_CSV
)

prediction_dataframe = pd.read_csv(
    CALIBRATED_PREDICTIONS_CSV
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


# Resolve paths from the clean test CSV.
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

prediction_dataframe["normalized_path"] = (
    prediction_dataframe["image_path"]
    .apply(normalize_path)
)


required_prediction_columns = [
    "normalized_path",
    "calibrated_confidence",
    "correct",
    "predicted_label",
]

missing_prediction_columns = [
    column
    for column in required_prediction_columns
    if column not in prediction_dataframe.columns
]

if missing_prediction_columns:
    raise ValueError(
        "The calibrated prediction file is missing:\n"
        f"{missing_prediction_columns}"
    )


prediction_dataframe[
    "calibrated_confidence"
] = pd.to_numeric(
    prediction_dataframe[
        "calibrated_confidence"
    ],
    errors="coerce",
)

prediction_dataframe["correct"] = (
    prediction_dataframe["correct"]
    .astype(str)
    .str.lower()
    .eq("true")
)


merged_dataframe = test_dataframe.merge(
    prediction_dataframe[
        [
            "normalized_path",
            "calibrated_confidence",
            "correct",
            "predicted_label",
        ]
    ],
    on="normalized_path",
    how="left",
)


missing_predictions = (
    merged_dataframe[
        "calibrated_confidence"
    ]
    .isna()
    .sum()
)

if missing_predictions > 0:
    raise RuntimeError(
        "Some test images could not be matched "
        "with calibrated predictions.\n"
        f"Missing predictions: {missing_predictions}"
    )


merged_dataframe["class_name"] = (
    merged_dataframe[LABEL_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)

print()
print(
    f"Successfully matched rows: "
    f"{len(merged_dataframe)}"
)


# ============================================================
# 7. Class baseline results
# ============================================================

class_baseline_results = []

for class_name in CLASS_NAMES:

    class_data = merged_dataframe[
        merged_dataframe["class_name"]
        == class_name
    ]

    class_correctness = (
        class_data["correct"]
        .astype(int)
        .values
    )

    class_confidence = (
        class_data["calibrated_confidence"]
        .values
    )

    class_accuracy = (
        class_correctness.mean()
    )

    class_mean_confidence = (
        class_confidence.mean()
    )

    class_gap = (
        class_mean_confidence
        - class_accuracy
    )

    class_ece = calculate_ece(
        confidence_values=class_confidence,
        correctness_values=class_correctness,
    )

    class_baseline_results.append(
        {
            "class_name": class_name,
            "sample_count": len(class_data),
            "accuracy": class_accuracy,
            "mean_confidence": (
                class_mean_confidence
            ),
            "confidence_accuracy_gap": (
                class_gap
            ),
            "expected_calibration_error": (
                class_ece
            ),
        }
    )


class_baseline_dataframe = pd.DataFrame(
    class_baseline_results
)

class_baseline_dataframe.to_csv(
    CLASS_RESULTS_PATH,
    index=False,
)


# ============================================================
# 8. Class-controlled attribute analysis
# ============================================================

group_results = []

for class_name in CLASS_NAMES:

    class_data = merged_dataframe[
        merged_dataframe["class_name"]
        == class_name
    ]

    class_baseline = (
        class_baseline_dataframe[
            class_baseline_dataframe[
                "class_name"
            ]
            == class_name
        ]
        .iloc[0]
    )

    for attribute_name in ATTRIBUTE_COLUMNS:

        for attribute_value in ATTRIBUTE_VALUES[
            attribute_name
        ]:

            group_data = class_data[
                class_data[attribute_name]
                .astype(str)
                .str.strip()
                .str.lower()
                == attribute_value
            ]

            group_size = len(group_data)

            if group_size == 0:
                continue

            group_correctness = (
                group_data["correct"]
                .astype(int)
                .values
            )

            group_confidence = (
                group_data[
                    "calibrated_confidence"
                ]
                .values
            )

            group_accuracy = (
                group_correctness.mean()
            )

            group_mean_confidence = (
                group_confidence.mean()
            )

            group_gap = (
                group_mean_confidence
                - group_accuracy
            )

            group_ece = calculate_ece(
                confidence_values=group_confidence,
                correctness_values=group_correctness,
            )

            accuracy_difference = (
                group_accuracy
                - class_baseline["accuracy"]
            )

            ece_difference = (
                group_ece
                - class_baseline[
                    "expected_calibration_error"
                ]
            )

            confidence_gap_difference = (
                group_gap
                - class_baseline[
                    "confidence_accuracy_gap"
                ]
            )

            if group_size < 30:
                group_size_warning = (
                    "small_group"
                )
            else:
                group_size_warning = (
                    "sufficient_for_screening"
                )

            group_results.append(
                {
                    "class_name": class_name,
                    "attribute": attribute_name,
                    "attribute_value": (
                        attribute_value
                    ),
                    "sample_count": group_size,

                    "group_accuracy": (
                        group_accuracy
                    ),
                    "class_accuracy": (
                        class_baseline[
                            "accuracy"
                        ]
                    ),
                    "accuracy_difference_vs_class": (
                        accuracy_difference
                    ),

                    "group_mean_confidence": (
                        group_mean_confidence
                    ),
                    "class_mean_confidence": (
                        class_baseline[
                            "mean_confidence"
                        ]
                    ),

                    "group_confidence_accuracy_gap": (
                        group_gap
                    ),
                    "class_confidence_accuracy_gap": (
                        class_baseline[
                            "confidence_accuracy_gap"
                        ]
                    ),
                    "confidence_gap_difference_vs_class": (
                        confidence_gap_difference
                    ),

                    "group_expected_calibration_error": (
                        group_ece
                    ),
                    "class_expected_calibration_error": (
                        class_baseline[
                            "expected_calibration_error"
                        ]
                    ),
                    "ece_difference_vs_class": (
                        ece_difference
                    ),

                    "group_size_warning": (
                        group_size_warning
                    ),
                }
            )


group_results_dataframe = pd.DataFrame(
    group_results
)

group_results_dataframe = (
    group_results_dataframe.sort_values(
        by=[
            "group_size_warning",
            "group_expected_calibration_error",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

group_results_dataframe.to_csv(
    GROUP_RESULTS_PATH,
    index=False,
)


# ============================================================
# 9. Select the main risk groups
# ============================================================

screening_results = (
    group_results_dataframe[
        group_results_dataframe["sample_count"]
        >= 30
    ]
    .copy()
)

screening_results["risk_score"] = (
    screening_results[
        "group_expected_calibration_error"
    ]
    + screening_results[
        "ece_difference_vs_class"
    ].clip(lower=0)
    + screening_results[
        "confidence_gap_difference_vs_class"
    ].clip(lower=0)
)

screening_results = (
    screening_results.sort_values(
        by=[
            "risk_score",
            "group_expected_calibration_error",
        ],
        ascending=[
            False,
            False,
        ],
    )
)

screening_results.to_csv(
    TOP_RISK_GROUPS_PATH,
    index=False,
)


# ============================================================
# 10. Print class baselines
# ============================================================

print()
print("=" * 70)
print("CLASS BASELINE RESULTS")
print("=" * 70)

print(
    class_baseline_dataframe.to_string(
        index=False
    )
)


# ============================================================
# 11. Print highest-risk class-controlled groups
# ============================================================

print()
print("=" * 70)
print("HIGHEST-RISK CLASS-CONTROLLED GROUPS")
print("=" * 70)

print(
    screening_results[
        [
            "class_name",
            "attribute",
            "attribute_value",
            "sample_count",
            "group_accuracy",
            "class_accuracy",
            "accuracy_difference_vs_class",
            "group_mean_confidence",
            "group_confidence_accuracy_gap",
            "group_expected_calibration_error",
            "ece_difference_vs_class",
        ]
    ].head(30).to_string(
        index=False
    )
)


# ============================================================
# 12. Plot top ECE groups
# ============================================================

top_ece_plot_data = (
    screening_results.head(25)
    .copy()
)

top_ece_plot_data["group_name"] = (
    top_ece_plot_data["class_name"]
    + " | "
    + top_ece_plot_data["attribute"]
    + "="
    + top_ece_plot_data["attribute_value"]
)

top_ece_plot_data = (
    top_ece_plot_data.sort_values(
        by="group_expected_calibration_error",
        ascending=True,
    )
)

plt.figure(
    figsize=(12, 10)
)

plt.barh(
    top_ece_plot_data["group_name"],
    top_ece_plot_data[
        "group_expected_calibration_error"
    ],
)

plt.xlabel(
    "Expected calibration error"
)

plt.ylabel(
    "WBC class and morphological value"
)

plt.title(
    "Top Class-Controlled Morphology Calibration Errors"
)

plt.tight_layout()

plt.savefig(
    ECE_PLOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 13. Plot accuracy differences versus class baseline
# ============================================================

accuracy_plot_data = (
    screening_results.copy()
)

accuracy_plot_data["group_name"] = (
    accuracy_plot_data["class_name"]
    + " | "
    + accuracy_plot_data["attribute"]
    + "="
    + accuracy_plot_data["attribute_value"]
)

accuracy_plot_data = (
    accuracy_plot_data.sort_values(
        by="accuracy_difference_vs_class",
        ascending=True,
    )
)

plt.figure(
    figsize=(12, 10)
)

plt.barh(
    accuracy_plot_data["group_name"],
    accuracy_plot_data[
        "accuracy_difference_vs_class"
    ],
)

plt.axvline(
    0,
    color="black",
    linewidth=1,
)

plt.xlabel(
    "Group accuracy minus class accuracy"
)

plt.ylabel(
    "WBC class and morphological value"
)

plt.title(
    "Accuracy Difference Relative to WBC-Class Baseline"
)

plt.tight_layout()

plt.savefig(
    ACCURACY_DIFF_PLOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 14. Final output
# ============================================================

print()
print("=" * 70)
print("CLASS-CONTROLLED ANALYSIS FINISHED")
print("=" * 70)

print()
print(
    f"Class baselines saved to:\n"
    f"{CLASS_RESULTS_PATH}"
)

print()
print(
    f"All class-controlled groups saved to:\n"
    f"{GROUP_RESULTS_PATH}"
)

print()
print(
    f"Top risk groups saved to:\n"
    f"{TOP_RISK_GROUPS_PATH}"
)

print()
print(
    f"ECE plot saved to:\n"
    f"{ECE_PLOT_PATH}"
)

print()
print(
    f"Accuracy difference plot saved to:\n"
    f"{ACCURACY_DIFF_PLOT_PATH}"
)
