from pathlib import Path

import numpy as np
import pandas as pd


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
    / "statistical_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "statistical_validation_results.csv"
)

RISK_GROUPS_PATH = (
    OUTPUT_DIR
    / "statistically_supported_risk_groups.csv"
)


# ============================================================
# 2. Definitions
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


# Groups below these sizes are not formally tested.
MIN_GROUP_SIZE = 30
MIN_REST_SIZE = 30

NUMBER_OF_BOOTSTRAP_SAMPLES = 500
NUMBER_OF_PERMUTATIONS = 500
SEED = 42


# ============================================================
# 3. Required-file checks
# ============================================================

if not CLEAN_TEST_CSV.exists():
    raise FileNotFoundError(
        f"Clean test CSV was not found:\n"
        f"{CLEAN_TEST_CSV}"
    )

if not CALIBRATED_PREDICTIONS_CSV.exists():
    raise FileNotFoundError(
        f"Calibrated prediction CSV was not found:\n"
        f"{CALIBRATED_PREDICTIONS_CSV}"
    )


# ============================================================
# 4. Path functions
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
# 5. ECE and metric functions
# ============================================================

def calculate_ece(
    confidence,
    correctness,
    number_of_bins=10,
):
    """
    Calculates expected calibration error.
    """

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

        lower_bound = (
            bin_index
            / number_of_bins
        )

        upper_bound = (
            bin_index + 1
        ) / number_of_bins

        if bin_index == number_of_bins - 1:

            in_bin = (
                (confidence >= lower_bound)
                & (confidence <= upper_bound)
            )

        else:

            in_bin = (
                (confidence >= lower_bound)
                & (confidence < upper_bound)
            )

        if not np.any(in_bin):
            continue

        bin_confidence = (
            confidence[in_bin].mean()
        )

        bin_accuracy = (
            correctness[in_bin].mean()
        )

        bin_fraction = (
            in_bin.sum()
            / len(confidence)
        )

        ece += (
            bin_fraction
            * abs(
                bin_confidence
                - bin_accuracy
            )
        )

    return float(ece)


def calculate_group_metrics(
    confidence,
    correctness,
):
    """
    Calculates the main metrics for one group.
    """

    confidence = np.asarray(
        confidence,
        dtype=float,
    )

    correctness = np.asarray(
        correctness,
        dtype=float,
    )

    accuracy = correctness.mean()
    mean_confidence = confidence.mean()

    confidence_accuracy_gap = (
        mean_confidence
        - accuracy
    )

    ece = calculate_ece(
        confidence=confidence,
        correctness=correctness,
    )

    return {
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "confidence_accuracy_gap": (
            confidence_accuracy_gap
        ),
        "ece": ece,
    }


# ============================================================
# 6. Bootstrap confidence intervals
# ============================================================

def bootstrap_difference(
    group_confidence,
    group_correctness,
    rest_confidence,
    rest_correctness,
):
    """
    Estimates 95 percent bootstrap confidence intervals
    for group minus rest differences.
    """

    rng = np.random.default_rng(
        SEED
    )

    group_confidence = np.asarray(
        group_confidence,
        dtype=float,
    )

    group_correctness = np.asarray(
        group_correctness,
        dtype=float,
    )

    rest_confidence = np.asarray(
        rest_confidence,
        dtype=float,
    )

    rest_correctness = np.asarray(
        rest_correctness,
        dtype=float,
    )

    group_size = len(
        group_confidence
    )

    rest_size = len(
        rest_confidence
    )

    accuracy_differences = []
    gap_differences = []
    ece_differences = []

    for _ in range(
        NUMBER_OF_BOOTSTRAP_SAMPLES
    ):

        group_indices = rng.integers(
            low=0,
            high=group_size,
            size=group_size,
        )

        rest_indices = rng.integers(
            low=0,
            high=rest_size,
            size=rest_size,
        )

        sampled_group_confidence = (
            group_confidence[group_indices]
        )

        sampled_group_correctness = (
            group_correctness[group_indices]
        )

        sampled_rest_confidence = (
            rest_confidence[rest_indices]
        )

        sampled_rest_correctness = (
            rest_correctness[rest_indices]
        )

        group_metrics = calculate_group_metrics(
            confidence=sampled_group_confidence,
            correctness=sampled_group_correctness,
        )

        rest_metrics = calculate_group_metrics(
            confidence=sampled_rest_confidence,
            correctness=sampled_rest_correctness,
        )

        accuracy_differences.append(
            group_metrics["accuracy"]
            - rest_metrics["accuracy"]
        )

        gap_differences.append(
            group_metrics[
                "confidence_accuracy_gap"
            ]
            - rest_metrics[
                "confidence_accuracy_gap"
            ]
        )

        ece_differences.append(
            group_metrics["ece"]
            - rest_metrics["ece"]
        )

    return {
        "accuracy_difference_ci_lower": (
            np.percentile(
                accuracy_differences,
                2.5,
            )
        ),

        "accuracy_difference_ci_upper": (
            np.percentile(
                accuracy_differences,
                97.5,
            )
        ),

        "gap_difference_ci_lower": (
            np.percentile(
                gap_differences,
                2.5,
            )
        ),

        "gap_difference_ci_upper": (
            np.percentile(
                gap_differences,
                97.5,
            )
        ),

        "ece_difference_ci_lower": (
            np.percentile(
                ece_differences,
                2.5,
            )
        ),

        "ece_difference_ci_upper": (
            np.percentile(
                ece_differences,
                97.5,
            )
        ),
    }


# ============================================================
# 7. Permutation tests
# ============================================================

def permutation_test(
    group_confidence,
    group_correctness,
    rest_confidence,
    rest_correctness,
):
    """
    Performs permutation tests for:

    1. Accuracy difference
    2. Confidence-gap difference
    3. ECE difference

    The group labels are shuffled within the same WBC class.
    """

    rng = np.random.default_rng(
        SEED
    )

    group_confidence = np.asarray(
        group_confidence,
        dtype=float,
    )

    group_correctness = np.asarray(
        group_correctness,
        dtype=float,
    )

    rest_confidence = np.asarray(
        rest_confidence,
        dtype=float,
    )

    rest_correctness = np.asarray(
        rest_correctness,
        dtype=float,
    )

    observed_group_metrics = (
        calculate_group_metrics(
            confidence=group_confidence,
            correctness=group_correctness,
        )
    )

    observed_rest_metrics = (
        calculate_group_metrics(
            confidence=rest_confidence,
            correctness=rest_correctness,
        )
    )

    observed_accuracy_difference = (
        observed_group_metrics["accuracy"]
        - observed_rest_metrics["accuracy"]
    )

    observed_gap_difference = (
        observed_group_metrics[
            "confidence_accuracy_gap"
        ]
        - observed_rest_metrics[
            "confidence_accuracy_gap"
        ]
    )

    observed_ece_difference = (
        observed_group_metrics["ece"]
        - observed_rest_metrics["ece"]
    )

    pooled_confidence = np.concatenate(
        [
            group_confidence,
            rest_confidence,
        ]
    )

    pooled_correctness = np.concatenate(
        [
            group_correctness,
            rest_correctness,
        ]
    )

    group_size = len(
        group_confidence
    )

    null_accuracy_differences = []
    null_gap_differences = []
    null_ece_differences = []

    for _ in range(
        NUMBER_OF_PERMUTATIONS
    ):

        shuffled_indices = rng.permutation(
            len(pooled_confidence)
        )

        shuffled_group_indices = (
            shuffled_indices[:group_size]
        )

        shuffled_rest_indices = (
            shuffled_indices[group_size:]
        )

        shuffled_group_metrics = (
            calculate_group_metrics(
                confidence=pooled_confidence[
                    shuffled_group_indices
                ],
                correctness=pooled_correctness[
                    shuffled_group_indices
                ],
            )
        )

        shuffled_rest_metrics = (
            calculate_group_metrics(
                confidence=pooled_confidence[
                    shuffled_rest_indices
                ],
                correctness=pooled_correctness[
                    shuffled_rest_indices
                ],
            )
        )

        null_accuracy_differences.append(
            shuffled_group_metrics["accuracy"]
            - shuffled_rest_metrics["accuracy"]
        )

        null_gap_differences.append(
            shuffled_group_metrics[
                "confidence_accuracy_gap"
            ]
            - shuffled_rest_metrics[
                "confidence_accuracy_gap"
            ]
        )

        null_ece_differences.append(
            shuffled_group_metrics["ece"]
            - shuffled_rest_metrics["ece"]
        )

    accuracy_p_value = (
        1
        + np.sum(
            np.abs(
                null_accuracy_differences
            )
            >= abs(
                observed_accuracy_difference
            )
        )
    ) / (
        NUMBER_OF_PERMUTATIONS + 1
    )

    gap_p_value = (
        1
        + np.sum(
            np.abs(
                null_gap_differences
            )
            >= abs(
                observed_gap_difference
            )
        )
    ) / (
        NUMBER_OF_PERMUTATIONS + 1
    )

    ece_p_value = (
        1
        + np.sum(
            np.abs(
                null_ece_differences
            )
            >= abs(
                observed_ece_difference
            )
        )
    ) / (
        NUMBER_OF_PERMUTATIONS + 1
    )

    return {
        "accuracy_p_value": accuracy_p_value,
        "gap_p_value": gap_p_value,
        "ece_p_value": ece_p_value,
    }


# ============================================================
# 8. Benjamini-Hochberg correction
# ============================================================

def benjamini_hochberg(p_values):
    """
    Applies Benjamini-Hochberg false-discovery-rate
    correction.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted_values = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(p_values)
    )[0]

    if len(valid_indices) == 0:
        return adjusted_values

    valid_p_values = p_values[
        valid_indices
    ]

    order = np.argsort(
        valid_p_values
    )

    sorted_p_values = valid_p_values[
        order
    ]

    number_of_tests = len(
        sorted_p_values
    )

    adjusted_sorted = (
        sorted_p_values
        * number_of_tests
        / np.arange(
            1,
            number_of_tests + 1,
        )
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    adjusted_valid = np.empty(
        number_of_tests,
        dtype=float,
    )

    adjusted_valid[
        order
    ] = adjusted_sorted

    adjusted_values[
        valid_indices
    ] = adjusted_valid

    return adjusted_values


# ============================================================
# 9. Load and merge predictions
# ============================================================

print("=" * 70)
print("STATISTICAL VALIDATION OF MORPHOLOGY RISK GROUPS")
print("=" * 70)

test_dataframe = pd.read_csv(
    CLEAN_TEST_CSV
)

prediction_dataframe = pd.read_csv(
    CALIBRATED_PREDICTIONS_CSV
)

print()
print(
    f"Test rows: "
    f"{len(test_dataframe)}"
)

print(
    f"Prediction rows: "
    f"{len(prediction_dataframe)}"
)


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
        ]
    ],
    on="normalized_path",
    how="left",
)

if (
    merged_dataframe[
        "calibrated_confidence"
    ]
    .isna()
    .any()
):
    raise RuntimeError(
        "Some test rows could not be matched "
        "with calibrated predictions."
    )

merged_dataframe["class_name"] = (
    merged_dataframe[LABEL_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)

print(
    f"Successfully matched rows: "
    f"{len(merged_dataframe)}"
)


# ============================================================
# 10. Calculate class-controlled results
# ============================================================

results = []

for class_name in CLASS_NAMES:

    class_data = merged_dataframe[
        merged_dataframe["class_name"]
        == class_name
    ].copy()

    class_confidence = (
        class_data[
            "calibrated_confidence"
        ]
        .values
    )

    class_correctness = (
        class_data["correct"]
        .astype(int)
        .values
    )

    class_metrics = calculate_group_metrics(
        confidence=class_confidence,
        correctness=class_correctness,
    )

    for attribute_name in ATTRIBUTE_COLUMNS:

        for attribute_value in ATTRIBUTE_VALUES[
            attribute_name
        ]:

            attribute_mask = (
                class_data[attribute_name]
                .astype(str)
                .str.strip()
                .str.lower()
                == attribute_value
            )

            group_data = class_data[
                attribute_mask
            ]

            rest_data = class_data[
                ~attribute_mask
            ]

            group_size = len(group_data)
            rest_size = len(rest_data)

            if group_size < MIN_GROUP_SIZE:
                continue

            if rest_size < MIN_REST_SIZE:
                continue

            group_confidence = (
                group_data[
                    "calibrated_confidence"
                ]
                .values
            )

            group_correctness = (
                group_data["correct"]
                .astype(int)
                .values
            )

            rest_confidence = (
                rest_data[
                    "calibrated_confidence"
                ]
                .values
            )

            rest_correctness = (
                rest_data["correct"]
                .astype(int)
                .values
            )

            group_metrics = (
                calculate_group_metrics(
                    confidence=group_confidence,
                    correctness=group_correctness,
                )
            )

            rest_metrics = (
                calculate_group_metrics(
                    confidence=rest_confidence,
                    correctness=rest_correctness,
                )
            )

            accuracy_difference = (
                group_metrics["accuracy"]
                - rest_metrics["accuracy"]
            )

            gap_difference = (
                group_metrics[
                    "confidence_accuracy_gap"
                ]
                - rest_metrics[
                    "confidence_accuracy_gap"
                ]
            )

            ece_difference = (
                group_metrics["ece"]
                - rest_metrics["ece"]
            )

            bootstrap_results = (
                bootstrap_difference(
                    group_confidence=group_confidence,
                    group_correctness=group_correctness,
                    rest_confidence=rest_confidence,
                    rest_correctness=rest_correctness,
                )
            )

            permutation_results = (
                permutation_test(
                    group_confidence=group_confidence,
                    group_correctness=group_correctness,
                    rest_confidence=rest_confidence,
                    rest_correctness=rest_correctness,
                )
            )

            results.append(
                {
                    "class_name": class_name,
                    "attribute": attribute_name,
                    "attribute_value": attribute_value,
                    "group_sample_count": group_size,
                    "rest_sample_count": rest_size,

                    "class_accuracy": (
                        class_metrics[
                            "accuracy"
                        ]
                    ),
                    "group_accuracy": (
                        group_metrics[
                            "accuracy"
                        ]
                    ),
                    "rest_accuracy": (
                        rest_metrics[
                            "accuracy"
                        ]
                    ),
                    "accuracy_difference_group_minus_rest": (
                        accuracy_difference
                    ),

                    "group_mean_confidence": (
                        group_metrics[
                            "mean_confidence"
                        ]
                    ),
                    "rest_mean_confidence": (
                        rest_metrics[
                            "mean_confidence"
                        ]
                    ),

                    "group_confidence_accuracy_gap": (
                        group_metrics[
                            "confidence_accuracy_gap"
                        ]
                    ),
                    "rest_confidence_accuracy_gap": (
                        rest_metrics[
                            "confidence_accuracy_gap"
                        ]
                    ),
                    "gap_difference_group_minus_rest": (
                        gap_difference
                    ),

                    "group_ece": (
                        group_metrics["ece"]
                    ),
                    "rest_ece": (
                        rest_metrics["ece"]
                    ),
                    "ece_difference_group_minus_rest": (
                        ece_difference
                    ),

                    "accuracy_difference_ci_lower": (
                        bootstrap_results[
                            "accuracy_difference_ci_lower"
                        ]
                    ),
                    "accuracy_difference_ci_upper": (
                        bootstrap_results[
                            "accuracy_difference_ci_upper"
                        ]
                    ),

                    "gap_difference_ci_lower": (
                        bootstrap_results[
                            "gap_difference_ci_lower"
                        ]
                    ),
                    "gap_difference_ci_upper": (
                        bootstrap_results[
                            "gap_difference_ci_upper"
                        ]
                    ),

                    "ece_difference_ci_lower": (
                        bootstrap_results[
                            "ece_difference_ci_lower"
                        ]
                    ),
                    "ece_difference_ci_upper": (
                        bootstrap_results[
                            "ece_difference_ci_upper"
                        ]
                    ),

                    "accuracy_p_value": (
                        permutation_results[
                            "accuracy_p_value"
                        ]
                    ),
                    "gap_p_value": (
                        permutation_results[
                            "gap_p_value"
                        ]
                    ),
                    "ece_p_value": (
                        permutation_results[
                            "ece_p_value"
                        ]
                    ),
                }
            )


results_dataframe = pd.DataFrame(
    results
)


# ============================================================
# 11. Apply false-discovery-rate correction
# ============================================================

results_dataframe[
    "accuracy_q_value"
] = benjamini_hochberg(
    results_dataframe[
        "accuracy_p_value"
    ].values
)

results_dataframe[
    "gap_q_value"
] = benjamini_hochberg(
    results_dataframe[
        "gap_p_value"
    ].values
)

results_dataframe[
    "ece_q_value"
] = benjamini_hochberg(
    results_dataframe[
        "ece_p_value"
    ].values
)


# ============================================================
# 12. Mark statistically supported risk groups
# ============================================================

results_dataframe[
    "accuracy_risk_supported"
] = (
    (results_dataframe[
        "accuracy_difference_group_minus_rest"
    ] < 0)
    & (
        results_dataframe[
            "accuracy_q_value"
        ] < 0.05
    )
)

results_dataframe[
    "overconfidence_risk_supported"
] = (
    (results_dataframe[
        "gap_difference_group_minus_rest"
    ] > 0)
    & (
        results_dataframe[
            "gap_q_value"
        ] < 0.05
    )
)

results_dataframe[
    "calibration_risk_supported"
] = (
    (results_dataframe[
        "ece_difference_group_minus_rest"
    ] > 0)
    & (
        results_dataframe[
            "ece_q_value"
        ] < 0.05
    )
)

results_dataframe[
    "statistically_supported_risk"
] = (
    results_dataframe[
        [
            "accuracy_risk_supported",
            "overconfidence_risk_supported",
            "calibration_risk_supported",
        ]
    ]
    .any(axis=1)
)


# Positive values represent worse risk.
results_dataframe["risk_score"] = (
    results_dataframe[
        "accuracy_difference_group_minus_rest"
    ]
    .mul(-1)
    .clip(lower=0)
    + results_dataframe[
        "gap_difference_group_minus_rest"
    ].clip(lower=0)
    + results_dataframe[
        "ece_difference_group_minus_rest"
    ].clip(lower=0)
)


results_dataframe = (
    results_dataframe.sort_values(
        by=[
            "statistically_supported_risk",
            "risk_score",
        ],
        ascending=[
            False,
            False,
        ],
    )
)


# ============================================================
# 13. Save all results
# ============================================================

results_dataframe.to_csv(
    RESULTS_PATH,
    index=False,
)

supported_risk_dataframe = (
    results_dataframe[
        results_dataframe[
            "statistically_supported_risk"
        ]
        == True
    ]
    .copy()
)

supported_risk_dataframe.to_csv(
    RISK_GROUPS_PATH,
    index=False,
)


# ============================================================
# 14. Print results
# ============================================================

print()
print("=" * 70)
print("TOP CLASS-CONTROLLED GROUPS")
print("=" * 70)

display_columns = [
    "class_name",
    "attribute",
    "attribute_value",
    "group_sample_count",
    "group_accuracy",
    "rest_accuracy",
    "accuracy_difference_group_minus_rest",
    "group_confidence_accuracy_gap",
    "rest_confidence_accuracy_gap",
    "gap_difference_group_minus_rest",
    "group_ece",
    "rest_ece",
    "ece_difference_group_minus_rest",
    "accuracy_q_value",
    "gap_q_value",
    "ece_q_value",
    "statistically_supported_risk",
]

print(
    results_dataframe[
        display_columns
    ].head(30).to_string(
        index=False
    )
)


print()
print("=" * 70)
print("STATISTICALLY SUPPORTED RISK GROUPS")
print("=" * 70)

if len(supported_risk_dataframe) == 0:

    print(
        "No class-controlled group passed the "
        "false-discovery-rate threshold of 0.05."
    )

else:

    print(
        supported_risk_dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )


print()
print("=" * 70)
print("STATISTICAL VALIDATION FINISHED")
print("=" * 70)

print()
print(
    f"All results saved to:\n"
    f"{RESULTS_PATH}"
)

print()
print(
    f"Supported risk groups saved to:\n"
    f"{RISK_GROUPS_PATH}"
)
