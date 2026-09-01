
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
    ALL_CASES_PATH,
    FAILURE_ANALYSIS_DIR,
    clean_prediction_dataframe,
    calculate_ece,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42
BOOTSTRAP_SAMPLES = 2000
ECE_BIN_COUNTS = [
    5,
    10,
    15,
]


BOOTSTRAP_OUTPUT_PATH = (
    FAILURE_ANALYSIS_DIR
    / "risk_group_bootstrap_summary.csv"
)

ECE_SENSITIVITY_OUTPUT_PATH = (
    FAILURE_ANALYSIS_DIR
    / "risk_group_ece_bin_sensitivity.csv"
)

BOOTSTRAP_PLOT_PATH = (
    FAILURE_ANALYSIS_DIR
    / "risk_group_bootstrap_distributions.png"
)


# ============================================================
# Load risk-group data
# ============================================================

if not ALL_CASES_PATH.exists():
    raise FileNotFoundError(
        f"Required file was not found:\n{ALL_CASES_PATH}"
    )

risk_group = pd.read_csv(
    ALL_CASES_PATH
)

risk_group = clean_prediction_dataframe(
    risk_group
)

required_columns = [
    "img_name",
    "correct",
    "calibrated_confidence",
]

missing_columns = [
    column
    for column in required_columns
    if column not in risk_group.columns
]

if missing_columns:
    raise KeyError(
        "Required columns are missing:\n"
        f"{missing_columns}"
    )

confidence = (
    risk_group[
        "calibrated_confidence"
    ]
    .to_numpy(
        dtype=float
    )
)

correctness = (
    risk_group[
        "correct"
    ]
    .to_numpy(
        dtype=bool
    )
)

sample_count = len(
    risk_group
)

if sample_count < 10:
    raise RuntimeError(
        "The risk group is too small for this analysis."
    )


# ============================================================
# Observed statistics
# ============================================================

observed_accuracy = (
    correctness.mean()
)

observed_mean_confidence = (
    confidence.mean()
)

observed_gap = (
    observed_mean_confidence
    - observed_accuracy
)

observed_ece_10 = calculate_ece(
    confidence=confidence,
    correctness=correctness,
    number_of_bins=10,
)

incorrect_count = int(
    (~correctness).sum()
)

high_confidence_error_count = int(
    (
        (~correctness)
        &
        (confidence >= 0.80)
    ).sum()
)


# ============================================================
# Bootstrap
# ============================================================

rng = np.random.default_rng(
    SEED
)

bootstrap_accuracy = []
bootstrap_mean_confidence = []
bootstrap_gap = []
bootstrap_ece = []

for _ in range(
    BOOTSTRAP_SAMPLES
):

    indices = rng.integers(
        low=0,
        high=sample_count,
        size=sample_count,
    )

    sample_confidence = (
        confidence[indices]
    )

    sample_correctness = (
        correctness[indices]
    )

    sample_accuracy = (
        sample_correctness.mean()
    )

    sample_mean_confidence = (
        sample_confidence.mean()
    )

    sample_gap = (
        sample_mean_confidence
        - sample_accuracy
    )

    sample_ece = calculate_ece(
        confidence=sample_confidence,
        correctness=sample_correctness,
        number_of_bins=10,
    )

    bootstrap_accuracy.append(
        sample_accuracy
    )

    bootstrap_mean_confidence.append(
        sample_mean_confidence
    )

    bootstrap_gap.append(
        sample_gap
    )

    bootstrap_ece.append(
        sample_ece
    )


def percentile_interval(values):
    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


accuracy_ci = percentile_interval(
    bootstrap_accuracy
)

confidence_ci = percentile_interval(
    bootstrap_mean_confidence
)

gap_ci = percentile_interval(
    bootstrap_gap
)

ece_ci = percentile_interval(
    bootstrap_ece
)


bootstrap_summary = pd.DataFrame([
    {
        "risk_group": (
            "monocyte_high_nuclear_cytoplasmic_ratio"
        ),
        "sample_count": sample_count,
        "correct_count": int(
            correctness.sum()
        ),
        "incorrect_count": incorrect_count,
        "high_confidence_error_count": (
            high_confidence_error_count
        ),
        "observed_accuracy": (
            observed_accuracy
        ),
        "accuracy_ci_lower": accuracy_ci[0],
        "accuracy_ci_upper": accuracy_ci[1],
        "observed_mean_confidence": (
            observed_mean_confidence
        ),
        "mean_confidence_ci_lower": (
            confidence_ci[0]
        ),
        "mean_confidence_ci_upper": (
            confidence_ci[1]
        ),
        "observed_confidence_accuracy_gap": (
            observed_gap
        ),
        "gap_ci_lower": gap_ci[0],
        "gap_ci_upper": gap_ci[1],
        "observed_ece_10_bins": (
            observed_ece_10
        ),
        "ece_ci_lower": ece_ci[0],
        "ece_ci_upper": ece_ci[1],
        "bootstrap_samples": (
            BOOTSTRAP_SAMPLES
        ),
    }
])

bootstrap_summary.to_csv(
    BOOTSTRAP_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# ECE bin-count sensitivity
# ============================================================

sensitivity_rows = []

for number_of_bins in ECE_BIN_COUNTS:

    ece_value = calculate_ece(
        confidence=confidence,
        correctness=correctness,
        number_of_bins=number_of_bins,
    )

    sensitivity_rows.append(
        {
            "risk_group": (
                "monocyte_high_nuclear_cytoplasmic_ratio"
            ),
            "number_of_bins": number_of_bins,
            "sample_count": sample_count,
            "ece": ece_value,
        }
    )

sensitivity_dataframe = pd.DataFrame(
    sensitivity_rows
)

sensitivity_dataframe.to_csv(
    ECE_SENSITIVITY_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Plot bootstrap distributions
# ============================================================

figure, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 4),
)

axes[0].hist(
    bootstrap_accuracy,
    bins=30,
    color="steelblue",
    edgecolor="black",
)

axes[0].axvline(
    observed_accuracy,
    color="red",
    linewidth=2,
)

axes[0].set_title(
    "Bootstrap accuracy"
)

axes[0].set_xlabel(
    "Accuracy"
)

axes[0].set_ylabel(
    "Frequency"
)

axes[1].hist(
    bootstrap_gap,
    bins=30,
    color="darkorange",
    edgecolor="black",
)

axes[1].axvline(
    observed_gap,
    color="red",
    linewidth=2,
)

axes[1].set_title(
    "Confidence-accuracy gap"
)

axes[1].set_xlabel(
    "Mean confidence minus accuracy"
)

axes[1].set_ylabel(
    "Frequency"
)

axes[2].hist(
    bootstrap_ece,
    bins=30,
    color="seagreen",
    edgecolor="black",
)

axes[2].axvline(
    observed_ece_10,
    color="red",
    linewidth=2,
)

axes[2].set_title(
    "Bootstrap ECE"
)

axes[2].set_xlabel(
    "ECE with 10 bins"
)

axes[2].set_ylabel(
    "Frequency"
)

figure.suptitle(
    "Bootstrap Validation of the Monocyte High-NCR Group",
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
    BOOTSTRAP_PLOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# Print results
# ============================================================

print("=" * 70)
print("RISK-GROUP ROBUSTNESS VALIDATION FINISHED")
print("=" * 70)

print()
print(
    f"Risk-group samples: {sample_count}"
)

print(
    f"Observed accuracy: "
    f"{observed_accuracy:.4f}"
)

print(
    f"95% bootstrap accuracy interval: "
    f"{accuracy_ci[0]:.4f} to {accuracy_ci[1]:.4f}"
)

print(
    f"Observed ECE with 10 bins: "
    f"{observed_ece_10:.4f}"
)

print(
    f"95% bootstrap ECE interval: "
    f"{ece_ci[0]:.4f} to {ece_ci[1]:.4f}"
)

print()
print(
    f"Bootstrap summary:\n"
    f"{BOOTSTRAP_OUTPUT_PATH}"
)

print()
print(
    f"ECE sensitivity table:\n"
    f"{ECE_SENSITIVITY_OUTPUT_PATH}"
)

print()
print(
    f"Bootstrap plot:\n"
    f"{BOOTSTRAP_PLOT_PATH}"
)

print()
print(
    "Important: bootstrap validation measures uncertainty "
    "within the current test group. It does not replace "
    "repeated independent train/test splits."
)
