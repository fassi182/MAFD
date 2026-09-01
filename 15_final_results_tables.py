

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sys.path.insert(
    0,
    "/content"
)

from mafd_utils import (
    FAILURE_ANALYSIS_DIR,
    ALL_CASES_PATH,
    FAILURE_CASES_PATH,
    CALIBRATION_SUMMARY_PATH,
    ATTRIBUTE_RESULTS_PATH,
    clean_prediction_dataframe,
    calculate_ece,
)


# ============================================================
# Output paths
# ============================================================

FINAL_RESULTS_DIR = (
    FAILURE_ANALYSIS_DIR
    / "final_results"
)

FINAL_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_GROUP_SUMMARY_PATH = (
    FINAL_RESULTS_DIR
    / "final_target_group_summary.csv"
)

FINAL_FAILURE_TABLE_PATH = (
    FINAL_RESULTS_DIR
    / "final_failure_case_table.csv"
)

FINAL_ATTRIBUTE_TABLE_PATH = (
    FINAL_RESULTS_DIR
    / "final_attribute_reliability_table.csv"
)

FINAL_METRICS_PATH = (
    FINAL_RESULTS_DIR
    / "final_overall_metrics.csv"
)

ECE_FIGURE_PATH = (
    FINAL_RESULTS_DIR
    / "top_morphology_groups_by_ece.png"
)

CONFIDENCE_FIGURE_PATH = (
    FINAL_RESULTS_DIR
    / "target_group_confidence_plot.png"
)


# ============================================================
# Load target-group data
# ============================================================

if not ALL_CASES_PATH.exists():
    raise FileNotFoundError(
        f"Missing target-group file:\n{ALL_CASES_PATH}"
    )

target_group = pd.read_csv(
    ALL_CASES_PATH
)

target_group = clean_prediction_dataframe(
    target_group
)

required_columns = [
    "img_name",
    "true_label",
    "predicted_label",
    "calibrated_confidence",
    "correct",
]

missing_columns = [
    column
    for column in required_columns
    if column not in target_group.columns
]

if missing_columns:
    raise KeyError(
        f"Missing target-group columns:\n{missing_columns}"
    )

confidence = (
    target_group[
        "calibrated_confidence"
    ].to_numpy(
        dtype=float
    )
)

correctness = (
    target_group[
        "correct"
    ].to_numpy(
        dtype=bool
    )
)

target_group_summary = pd.DataFrame([
    {
        "group": (
            "monocyte + "
            "nuclear_cytoplasmic_ratio = high"
        ),
        "sample_count": len(
            target_group
        ),
        "correct_count": int(
            correctness.sum()
        ),
        "incorrect_count": int(
            (~correctness).sum()
        ),
        "accuracy": correctness.mean(),
        "mean_calibrated_confidence": (
            confidence.mean()
        ),
        "confidence_accuracy_gap": (
            confidence.mean()
            - correctness.mean()
        ),
        "ece_5_bins": calculate_ece(
            confidence=confidence,
            correctness=correctness,
            number_of_bins=5,
        ),
        "ece_10_bins": calculate_ece(
            confidence=confidence,
            correctness=correctness,
            number_of_bins=10,
        ),
        "ece_15_bins": calculate_ece(
            confidence=confidence,
            correctness=correctness,
            number_of_bins=15,
        ),
    }
])

target_group_summary.to_csv(
    FINAL_GROUP_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Final failure-case table
# ============================================================

if FAILURE_CASES_PATH.exists():

    failure_cases = pd.read_csv(
        FAILURE_CASES_PATH
    )

    failure_cases = clean_prediction_dataframe(
        failure_cases
    )

    failure_columns = [
        "img_name",
        "true_label",
        "predicted_label",
        "calibrated_confidence",
        "cell_size",
        "cell_shape",
        "nucleus_shape",
        "nuclear_cytoplasmic_ratio",
        "cytoplasm_texture",
        "cytoplasm_colour",
        "resolved_path",
    ]

    failure_columns = [
        column
        for column in failure_columns
        if column in failure_cases.columns
    ]

    final_failure_table = (
        failure_cases[
            failure_columns
        ]
        .sort_values(
            by="calibrated_confidence",
            ascending=False,
        )
    )

    final_failure_table.to_csv(
        FINAL_FAILURE_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Copy and sort all morphology-group results
# ============================================================

if ATTRIBUTE_RESULTS_PATH.exists():

    attribute_results = pd.read_csv(
        ATTRIBUTE_RESULTS_PATH
    )

    ece_column = (
        "calibrated_expected_calibration_error"
    )

    if ece_column in attribute_results.columns:

        attribute_results = (
            attribute_results
            .sort_values(
                by=ece_column,
                ascending=False,
            )
        )

    attribute_results.to_csv(
        FINAL_ATTRIBUTE_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Copy overall calibration metrics
# ============================================================

if CALIBRATION_SUMMARY_PATH.exists():

    calibration_summary = pd.read_csv(
        CALIBRATION_SUMMARY_PATH
    )

    calibration_summary.to_csv(
        FINAL_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Figure 1: Top morphology groups by ECE
# ============================================================

if ATTRIBUTE_RESULTS_PATH.exists():

    attribute_results = pd.read_csv(
        ATTRIBUTE_RESULTS_PATH
    )

    ece_column = (
        "calibrated_expected_calibration_error"
    )

    if ece_column in attribute_results.columns:

        plot_dataframe = (
            attribute_results
            .sort_values(
                by=ece_column,
                ascending=False,
            )
            .head(15)
            .copy()
        )

        plot_dataframe["group_name"] = (
            plot_dataframe["attribute"]
            .astype(str)
            + " = "
            + plot_dataframe["attribute_value"]
            .astype(str)
        )

        colors = []

        for _, row in plot_dataframe.iterrows():

            is_target_group = (
                str(row["attribute"])
                == "nuclear_cytoplasmic_ratio"
                and
                str(row["attribute_value"])
                == "high"
            )

            colors.append(
                "red"
                if is_target_group
                else "steelblue"
            )

        plt.figure(
            figsize=(10, 7)
        )

        plt.barh(
            plot_dataframe["group_name"],
            plot_dataframe[ece_column],
            color=colors,
        )

        plt.gca().invert_yaxis()

        plt.xlabel(
            "Calibrated Expected Calibration Error"
        )

        plt.ylabel(
            "Morphological group"
        )

        plt.title(
            "Morphological Groups Ranked by Calibration Error"
        )

        plt.tight_layout()

        plt.savefig(
            ECE_FIGURE_PATH,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# Figure 2: Confidence and correctness in target group
# ============================================================

plot_dataframe = target_group.copy()

plot_dataframe["case_index"] = np.arange(
    len(plot_dataframe)
)

plot_dataframe["status"] = (
    plot_dataframe["correct"]
    .map(
        {
            True: "Correct",
            False: "Failure",
        }
    )
)

colors = plot_dataframe["correct"].map(
    {
        True: "seagreen",
        False: "red",
    }
)

plt.figure(
    figsize=(10, 5)
)

plt.scatter(
    plot_dataframe["case_index"],
    plot_dataframe["calibrated_confidence"],
    c=colors,
    s=80,
    edgecolors="black",
)

plt.axhline(
    0.80,
    color="gray",
    linestyle="--",
    linewidth=1,
    label="0.80 confidence",
)

plt.xlabel(
    "Case index"
)

plt.ylabel(
    "Calibrated confidence"
)

plt.title(
    "Confidence for Monocytes with High "
    "Nuclear-Cytoplasmic Ratio"
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CONFIDENCE_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Print final outputs
# ============================================================

print("=" * 70)
print("FINAL RESULTS FILES CREATED")
print("=" * 70)

print()
print(
    f"Target-group summary:\n"
    f"{FINAL_GROUP_SUMMARY_PATH}"
)

print()
print(
    f"Final failure table:\n"
    f"{FINAL_FAILURE_TABLE_PATH}"
)

print()
print(
    f"Final attribute reliability table:\n"
    f"{FINAL_ATTRIBUTE_TABLE_PATH}"
)

print()
print(
    f"Final overall metrics:\n"
    f"{FINAL_METRICS_PATH}"
)

print()
print(
    f"ECE figure:\n"
    f"{ECE_FIGURE_PATH}"
)

print()
print(
    f"Target-group confidence figure:\n"
    f"{CONFIDENCE_FIGURE_PATH}"
)
