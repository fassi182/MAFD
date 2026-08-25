from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 1. Paths
# ============================================================

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
    / "failure_case_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ALL_RISK_GROUP_PATH = (
    OUTPUT_DIR
    / "monocyte_high_ncr_all_cases.csv"
)

FAILURE_CASES_PATH = (
    OUTPUT_DIR
    / "monocyte_high_ncr_failure_cases.csv"
)

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "monocyte_high_ncr_failure_contact_sheet.png"
)


# ============================================================
# 2. Dataset paths
# ============================================================

DATASET_ROOT = Path(
    "/content/WBCAtt"
)

IMAGE_ROOT = (
    DATASET_ROOT
    / "PBC_dataset_normal_DIB"
)


# ============================================================
# 3. Check required files
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
# 4. Resolve image paths
# ============================================================

def resolve_image_path(raw_value, label):
    """
    Resolves image paths from absolute paths, relative paths,
    filenames, and class folders.
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
# 5. Load data
# ============================================================

print("=" * 70)
print("EXTRACTING FAILURE CASES")
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


# ============================================================
# 6. Prepare test paths
# ============================================================

test_dataframe["normalized_path"] = (
    test_dataframe.apply(
        lambda row: normalize_path(
            resolve_image_path(
                raw_value=row["path"],
                label=str(
                    row["label"]
                ).strip().lower(),
            )
        ),
        axis=1,
    )
)

prediction_dataframe["normalized_path"] = (
    prediction_dataframe["image_path"]
    .apply(normalize_path)
)


# ============================================================
# 7. Prepare prediction columns
# ============================================================

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


# ============================================================
# 8. Merge test attributes with predictions
# ============================================================

prediction_columns = [
    "normalized_path",
    "true_label",
    "predicted_label",
    "calibrated_confidence",
    "correct",
]

merged_dataframe = test_dataframe.merge(
    prediction_dataframe[
        prediction_columns
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
        "Some images could not be matched "
        "with predictions.\n"
        f"Missing predictions: {missing_predictions}"
    )


# ============================================================
# 9. Select the supported risk group
# ============================================================

merged_dataframe["normalized_label"] = (
    merged_dataframe["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

merged_dataframe[
    "normalized_ncr"
] = (
    merged_dataframe[
        "nuclear_cytoplasmic_ratio"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)

risk_group_dataframe = merged_dataframe[
    (
        merged_dataframe[
            "normalized_label"
        ]
        == "monocyte"
    )
    &
    (
        merged_dataframe[
            "normalized_ncr"
        ]
        == "high"
    )
].copy()


if len(risk_group_dataframe) == 0:
    raise RuntimeError(
        "No cases were found for:\n"
        "monocyte + nuclear_cytoplasmic_ratio = high"
    )


# ============================================================
# 10. Add review columns
# ============================================================

risk_group_dataframe[
    "review_status"
] = ""

risk_group_dataframe[
    "visual_ambiguity"
] = ""

risk_group_dataframe[
    "error_explanation"
] = ""

risk_group_dataframe[
    "manual_notes"
] = ""


risk_group_dataframe[
    "failure_type"
] = risk_group_dataframe[
    "correct"
].apply(
    lambda value: (
        "correct_prediction"
        if value
        else "classification_failure"
    )
)


# Incorrect predictions first, then highest confidence.
risk_group_dataframe = (
    risk_group_dataframe.sort_values(
        by=[
            "correct",
            "calibrated_confidence",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)


# ============================================================
# 11. Save all risk-group cases
# ============================================================

columns_to_save = [
    "img_name",
    "path",
    "resolved_path",
    "label",
    "true_label",
    "predicted_label",
    "calibrated_confidence",
    "correct",
    "failure_type",
] + [
    column
    for column in [
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
    if column in risk_group_dataframe.columns
] + [
    "review_status",
    "visual_ambiguity",
    "error_explanation",
    "manual_notes",
]


risk_group_dataframe[
    columns_to_save
].to_csv(
    ALL_RISK_GROUP_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. Save only incorrect cases
# ============================================================

failure_cases_dataframe = (
    risk_group_dataframe[
        risk_group_dataframe["correct"]
        == False
    ]
    .copy()
)

failure_cases_dataframe[
    columns_to_save
].to_csv(
    FAILURE_CASES_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 13. Print failure cases
# ============================================================

print()
print("=" * 70)
print("RISK GROUP SUMMARY")
print("=" * 70)

print(
    f"Total risk-group cases: "
    f"{len(risk_group_dataframe)}"
)

print(
    f"Incorrect predictions: "
    f"{len(failure_cases_dataframe)}"
)

print(
    f"Correct predictions: "
    f"{len(risk_group_dataframe) - len(failure_cases_dataframe)}"
)

print()
print("=" * 70)
print("INCORRECT PREDICTIONS TO REVIEW")
print("=" * 70)

if len(failure_cases_dataframe) == 0:

    print(
        "No incorrect predictions were found "
        "in this risk group."
    )

else:

    print(
        failure_cases_dataframe[
            [
                "img_name",
                "true_label",
                "predicted_label",
                "calibrated_confidence",
                "nuclear_cytoplasmic_ratio",
                "cell_size",
                "cell_shape",
                "nucleus_shape",
                "cytoplasm_texture",
                "cytoplasm_colour",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 14. Create a contact sheet for failures
# ============================================================

if len(failure_cases_dataframe) > 0:

    number_of_rows = len(
        failure_cases_dataframe
    )

    figure, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=1,
        figsize=(
            8,
            5 * number_of_rows,
        ),
    )

    if number_of_rows == 1:
        axes = [axes]

    for row_index, (_, row) in enumerate(
        failure_cases_dataframe.iterrows()
    ):

        axis = axes[row_index]

        image = Image.open(
            row["resolved_path"]
        ).convert("RGB")

        axis.imshow(image)
        axis.axis("off")

        title = (
            f"Image: {row['img_name']}\n"
            f"True: {row['true_label']} | "
            f"Predicted: {row['predicted_label']}\n"
            f"Confidence: "
            f"{row['calibrated_confidence']:.4f}\n"
            f"Monocyte | "
            f"Nuclear-cytoplasmic ratio: high"
        )

        axis.set_title(
            title,
            fontsize=11,
            color="red",
        )

    figure.suptitle(
        "Failure Cases: Monocyte with High "
        "Nuclear-Cytoplasmic Ratio",
        fontsize=15,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    figure.savefig(
        CONTACT_SHEET_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print()
    print(
        f"Failure contact sheet saved to:\n"
        f"{CONTACT_SHEET_PATH}"
    )


# ============================================================
# 15. Final output
# ============================================================

print()
print("=" * 70)
print("FAILURE-CASE EXTRACTION FINISHED")
print("=" * 70)

print()
print(
    f"All risk-group cases saved to:\n"
    f"{ALL_RISK_GROUP_PATH}"
)

print()
print(
    f"Failure cases saved to:\n"
    f"{FAILURE_CASES_PATH}"
)

if len(failure_cases_dataframe) > 0:
    print()
    print(
        f"Contact sheet saved to:\n"
        f"{CONTACT_SHEET_PATH}"
    )
