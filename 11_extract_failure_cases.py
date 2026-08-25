# ============================================================
# 11_extract_failure_cases.py
#
# Extract failure cases from:
# monocyte + nuclear_cytoplasmic_ratio = high
# ============================================================

from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# ============================================================
# 1. Configuration
# ============================================================

PRIMARY_MODEL_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/MAFD_OUTPUTS/model_a_clean"
)

FALLBACK_MODEL_OUTPUT_DIR = Path(
    "/content/MAFD_OUTPUTS/model_a_clean"
)

DATASET_ROOT = Path(
    "/content/WBCAtt"
)

IMAGE_ROOT = DATASET_ROOT / "PBC_dataset_normal_DIB"


# ============================================================
# 2. Select available model-output directory
# ============================================================

if PRIMARY_MODEL_OUTPUT_DIR.exists():
    MODEL_OUTPUT_DIR = PRIMARY_MODEL_OUTPUT_DIR
elif FALLBACK_MODEL_OUTPUT_DIR.exists():
    MODEL_OUTPUT_DIR = FALLBACK_MODEL_OUTPUT_DIR
else:
    raise FileNotFoundError(
        "Could not find the model output directory.\n"
        f"Checked:\n"
        f"1. {PRIMARY_MODEL_OUTPUT_DIR}\n"
        f"2. {FALLBACK_MODEL_OUTPUT_DIR}"
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
    exist_ok=True
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
# 3. Check required files and folders
# ============================================================

if not CLEAN_TEST_CSV.exists():
    raise FileNotFoundError(
        f"Clean test CSV was not found:\n{CLEAN_TEST_CSV}"
    )

if not CALIBRATED_PREDICTIONS_CSV.exists():
    raise FileNotFoundError(
        "Calibrated prediction CSV was not found:\n"
        f"{CALIBRATED_PREDICTIONS_CSV}"
    )

if not IMAGE_ROOT.exists():
    raise FileNotFoundError(
        f"Image directory was not found:\n{IMAGE_ROOT}"
    )


# ============================================================
# 4. Utility functions
# ============================================================

def normalize_text(value):
    """
    Converts a value into a clean lowercase string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_path(path):
    """
    Converts a path to an absolute normalized lowercase path.
    """
    return str(
        Path(path).resolve()
    ).replace("\\", "/").lower()


def parse_boolean(value):
    """
    Converts common CSV representations into True or False.
    """
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "correct",
        "right"
    }

    false_values = {
        "false",
        "0",
        "no",
        "n",
        "incorrect",
        "wrong"
    }

    if text in true_values:
        return True

    if text in false_values:
        return False

    raise ValueError(
        f"Could not interpret this value as True or False: {value}"
    )


def build_image_index():
    """
    Builds an index of all images in the dataset.

    The index allows the script to resolve images even when the CSV
    contains only a filename instead of a complete path.
    """
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    image_index = {}

    for image_path in IMAGE_ROOT.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in valid_extensions:
            continue

        filename_key = image_path.name.strip().lower()

        if filename_key not in image_index:
            image_index[filename_key] = []

        image_index[filename_key].append(
            image_path.resolve()
        )

    if len(image_index) == 0:
        raise RuntimeError(
            f"No image files were found inside:\n{IMAGE_ROOT}"
        )

    return image_index


def resolve_image_path(
    raw_value,
    image_index,
    label=None
):
    """
    Resolves an image path from:

    1. An absolute existing path.
    2. A relative existing path.
    3. Dataset-relative paths.
    4. The expected class folder.
    5. The image filename index.
    """

    if pd.isna(raw_value):
        raise FileNotFoundError(
            "The image path value is empty."
        )

    raw_text = str(raw_value).strip()

    if raw_text == "":
        raise FileNotFoundError(
            "The image path value is empty."
        )

    raw_text = raw_text.replace("\\", "/")
    raw_path = Path(raw_text)

    candidates = []

    # Direct path from the CSV.
    candidates.append(raw_path)

    # Absolute path from the CSV.
    if raw_path.is_absolute():
        candidates.append(raw_path)

    # Paths relative to common dataset locations.
    candidates.extend([
        DATASET_ROOT / raw_path,
        IMAGE_ROOT / raw_path,
        DATASET_ROOT / raw_path.name,
        IMAGE_ROOT / raw_path.name,
    ])

    # Expected class-folder paths.
    if label is not None:
        label_text = str(label).strip()

        label_variants = {
            label_text,
            label_text.lower(),
            label_text.capitalize(),
            label_text.title(),
        }

        for label_variant in label_variants:
            candidates.extend([
                IMAGE_ROOT / label_variant / raw_path,
                IMAGE_ROOT / label_variant / raw_path.name,
            ])

    checked_paths = set()

    for candidate in candidates:

        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        candidate_key = str(candidate).lower()

        if candidate_key in checked_paths:
            continue

        checked_paths.add(candidate_key)

        if candidate.exists() and candidate.is_file():
            return candidate

    # Filename-based fallback.
    filename_key = raw_path.name.strip().lower()

    matching_paths = image_index.get(
        filename_key,
        []
    )

    if len(matching_paths) == 1:
        return matching_paths[0]

    if len(matching_paths) > 1 and label is not None:

        label_text = str(label).strip().lower()

        label_matches = [
            path
            for path in matching_paths
            if path.parent.name.strip().lower()
            == label_text
        ]

        if len(label_matches) == 1:
            return label_matches[0]

        raise FileNotFoundError(
            "The filename exists in multiple class folders, "
            "but the correct image could not be identified.\n"
            f"CSV value: {raw_value}\n"
            f"Label: {label}\n"
            f"Matching paths: {matching_paths}"
        )

    raise FileNotFoundError(
        "Could not resolve image path.\n"
        f"CSV value: {raw_value}\n"
        f"Label: {label}"
    )


def find_column(dataframe, possible_names, description):
    """
    Finds the first available column from a list of possible names.
    """
    for column_name in possible_names:
        if column_name in dataframe.columns:
            return column_name

    raise KeyError(
        f"Could not find the {description} column.\n"
        f"Expected one of: {possible_names}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


# ============================================================
# 5. Load the CSV files
# ============================================================

print("=" * 70)
print("EXTRACTING FAILURE CASES")
print("=" * 70)

print()
print(f"Model output directory:")
print(MODEL_OUTPUT_DIR)

print()
print(f"Clean test CSV:")
print(CLEAN_TEST_CSV)

print()
print(f"Calibrated predictions CSV:")
print(CALIBRATED_PREDICTIONS_CSV)

test_dataframe = pd.read_csv(
    CLEAN_TEST_CSV
)

prediction_dataframe = pd.read_csv(
    CALIBRATED_PREDICTIONS_CSV
)

print()
print(f"Clean test rows: {len(test_dataframe)}")
print(f"Prediction rows: {len(prediction_dataframe)}")


# ============================================================
# 6. Check the main test-data columns
# ============================================================

required_test_columns = [
    "path",
    "label",
]

missing_test_columns = [
    column
    for column in required_test_columns
    if column not in test_dataframe.columns
]

if missing_test_columns:
    raise KeyError(
        "The clean test CSV is missing these columns:\n"
        f"{missing_test_columns}\n\n"
        f"Available columns:\n{list(test_dataframe.columns)}"
    )


# ============================================================
# 7. Identify prediction-data columns
# ============================================================

prediction_path_column = find_column(
    prediction_dataframe,
    [
        "image_path",
        "path",
        "img_path",
        "filepath",
        "file_path",
    ],
    "prediction image-path"
)

prediction_true_label_column = find_column(
    prediction_dataframe,
    [
        "true_label",
        "label",
        "target",
        "actual_label",
    ],
    "prediction true-label"
)

prediction_label_column = find_column(
    prediction_dataframe,
    [
        "predicted_label",
        "prediction",
        "predicted_class",
        "pred_label",
    ],
    "prediction predicted-label"
)

prediction_confidence_column = find_column(
    prediction_dataframe,
    [
        "calibrated_confidence",
        "confidence",
        "calibrated_probability",
        "max_calibrated_probability",
    ],
    "prediction confidence"
)


# ============================================================
# 8. Build an image index
# ============================================================

print()
print("Indexing dataset images...")

image_index = build_image_index()

print(
    f"Indexed unique image filenames: {len(image_index)}"
)


# ============================================================
# 9. Resolve test image paths
#
# This section fixes the original error.
# The resolved_path column is explicitly created here.
# ============================================================

print()
print("Resolving clean-test image paths...")

resolved_test_paths = []

for row_number, row in test_dataframe.iterrows():

    resolved_path = resolve_image_path(
        raw_value=row["path"],
        label=row["label"],
        image_index=image_index
    )

    resolved_test_paths.append(
        str(resolved_path)
    )

    if (row_number + 1) % 500 == 0:
        print(
            f"Resolved {row_number + 1} test images..."
        )


# This column must exist before the merge.
test_dataframe["resolved_path"] = resolved_test_paths

test_dataframe["normalized_path"] = (
    test_dataframe["resolved_path"]
    .apply(normalize_path)
)


# ============================================================
# 10. Resolve prediction image paths
# ============================================================

print()
print("Resolving prediction image paths...")

resolved_prediction_paths = []

for row_number, row in prediction_dataframe.iterrows():

    label_value = row[prediction_true_label_column]

    resolved_path = resolve_image_path(
        raw_value=row[prediction_path_column],
        label=label_value,
        image_index=image_index
    )

    resolved_prediction_paths.append(
        str(resolved_path)
    )

    if (row_number + 1) % 500 == 0:
        print(
            f"Resolved {row_number + 1} prediction images..."
        )

prediction_dataframe["resolved_prediction_path"] = (
    resolved_prediction_paths
)

prediction_dataframe["normalized_path"] = (
    prediction_dataframe["resolved_prediction_path"]
    .apply(normalize_path)
)


# ============================================================
# 11. Prepare prediction columns
# ============================================================

prediction_dataframe["true_label"] = (
    prediction_dataframe[
        prediction_true_label_column
    ]
    .astype(str)
    .str.strip()
)

prediction_dataframe["predicted_label"] = (
    prediction_dataframe[
        prediction_label_column
    ]
    .astype(str)
    .str.strip()
)

prediction_dataframe["calibrated_confidence"] = pd.to_numeric(
    prediction_dataframe[
        prediction_confidence_column
    ],
    errors="coerce"
)

if prediction_dataframe[
    "calibrated_confidence"
].isna().any():

    invalid_count = prediction_dataframe[
        "calibrated_confidence"
    ].isna().sum()

    raise ValueError(
        "Some prediction confidence values could not be converted "
        f"to numbers. Invalid values: {invalid_count}"
    )


# Use the existing correct column if available.
if "correct" in prediction_dataframe.columns:

    prediction_dataframe["correct"] = (
        prediction_dataframe["correct"]
        .apply(parse_boolean)
    )

else:

    prediction_dataframe["correct"] = (
        prediction_dataframe["true_label"]
        .str.lower()
        ==
        prediction_dataframe["predicted_label"]
        .str.lower()
    )


# ============================================================
# 12. Check duplicate image keys
# ============================================================

duplicate_test_paths = (
    test_dataframe["normalized_path"]
    .duplicated()
    .sum()
)

duplicate_prediction_paths = (
    prediction_dataframe["normalized_path"]
    .duplicated()
    .sum()
)

if duplicate_test_paths > 0:
    raise RuntimeError(
        "The clean test CSV contains duplicate image paths:\n"
        f"Duplicate rows: {duplicate_test_paths}"
    )

if duplicate_prediction_paths > 0:
    raise RuntimeError(
        "The prediction CSV contains duplicate image paths:\n"
        f"Duplicate rows: {duplicate_prediction_paths}"
    )


# ============================================================
# 13. Merge clean annotations with predictions
# ============================================================

prediction_columns_for_merge = [
    "normalized_path",
    "true_label",
    "predicted_label",
    "calibrated_confidence",
    "correct",
]

merged_dataframe = test_dataframe.merge(
    prediction_dataframe[
        prediction_columns_for_merge
    ],
    on="normalized_path",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 14. Confirm that every test row has a prediction
# ============================================================

missing_prediction_mask = (
    merged_dataframe["predicted_label"]
    .isna()
)

missing_prediction_count = (
    missing_prediction_mask.sum()
)

if missing_prediction_count > 0:

    missing_rows = merged_dataframe[
        missing_prediction_mask
    ][
        [
            "img_name",
            "path",
            "resolved_path",
            "normalized_path",
        ]
    ]

    print()
    print("Images without matching predictions:")
    print(missing_rows.to_string(index=False))

    raise RuntimeError(
        "Some images could not be matched with predictions.\n"
        f"Missing predictions: {missing_prediction_count}"
    )


# ============================================================
# 15. Normalize morphology attributes
# ============================================================

merged_dataframe["normalized_label"] = (
    merged_dataframe["label"]
    .apply(normalize_text)
)

merged_dataframe["normalized_ncr"] = (
    merged_dataframe[
        "nuclear_cytoplasmic_ratio"
    ]
    .apply(normalize_text)
)


# ============================================================
# 16. Select the supported risk group
# ============================================================

risk_group_mask = (
    (
        merged_dataframe["normalized_label"]
        == "monocyte"
    )
    &
    (
        merged_dataframe["normalized_ncr"]
        == "high"
    )
)

risk_group_dataframe = (
    merged_dataframe[
        risk_group_mask
    ]
    .copy()
)

if len(risk_group_dataframe) == 0:
    raise RuntimeError(
        "No rows were found for:\n"
        "label = monocyte\n"
        "nuclear_cytoplasmic_ratio = high\n\n"
        "Check the actual values in these columns:\n"
        f"Labels: {sorted(merged_dataframe['normalized_label'].unique())}\n"
        f"NCR values: {sorted(merged_dataframe['normalized_ncr'].unique())}"
    )


# ============================================================
# 17. Add review columns
# ============================================================

risk_group_dataframe["review_status"] = ""
risk_group_dataframe["visual_ambiguity"] = ""
risk_group_dataframe["error_explanation"] = ""
risk_group_dataframe["manual_notes"] = ""

risk_group_dataframe["failure_type"] = (
    risk_group_dataframe["correct"]
    .apply(
        lambda value: (
            "correct_prediction"
            if value
            else "classification_failure"
        )
    )
)


# Incorrect predictions appear first.
# Within each group, high-confidence cases appear first.
risk_group_dataframe = (
    risk_group_dataframe
    .sort_values(
        by=[
            "correct",
            "calibrated_confidence",
        ],
        ascending=[
            True,
            False,
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 18. Select columns for export
# ============================================================

base_columns_to_save = [
    "img_name",
    "path",
    "resolved_path",
    "label",
    "true_label",
    "predicted_label",
    "calibrated_confidence",
    "correct",
    "failure_type",
]

morphology_columns = [
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

review_columns = [
    "review_status",
    "visual_ambiguity",
    "error_explanation",
    "manual_notes",
]

columns_to_save = []

for column in (
    base_columns_to_save
    + morphology_columns
    + review_columns
):

    if column in risk_group_dataframe.columns:
        columns_to_save.append(column)


# Confirm that resolved_path was successfully retained.
required_export_columns = [
    "img_name",
    "path",
    "resolved_path",
    "label",
    "true_label",
    "predicted_label",
    "calibrated_confidence",
    "correct",
    "failure_type",
]

missing_export_columns = [
    column
    for column in required_export_columns
    if column not in risk_group_dataframe.columns
]

if missing_export_columns:
    raise KeyError(
        "Required export columns are missing:\n"
        f"{missing_export_columns}\n\n"
        f"Available columns:\n"
        f"{list(risk_group_dataframe.columns)}"
    )


# ============================================================
# 19. Extract incorrect predictions
# ============================================================

failure_cases_dataframe = (
    risk_group_dataframe[
        risk_group_dataframe["correct"] == False
    ]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 20. Save all risk-group cases
# ============================================================

risk_group_dataframe[
    columns_to_save
].to_csv(
    ALL_RISK_GROUP_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. Save only incorrect risk-group cases
# ============================================================

failure_cases_dataframe[
    columns_to_save
].to_csv(
    FAILURE_CASES_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 22. Print summary
# ============================================================

total_risk_group_cases = len(
    risk_group_dataframe
)

total_failure_cases = len(
    failure_cases_dataframe
)

total_correct_cases = (
    total_risk_group_cases
    - total_failure_cases
)

risk_group_accuracy = (
    total_correct_cases
    / total_risk_group_cases
)

print()
print("=" * 70)
print("RISK GROUP SUMMARY")
print("=" * 70)

print()
print(
    "Risk group: "
    "monocyte + nuclear_cytoplasmic_ratio = high"
)

print(
    f"Total risk-group cases: "
    f"{total_risk_group_cases}"
)

print(
    f"Correct predictions: "
    f"{total_correct_cases}"
)

print(
    f"Incorrect predictions: "
    f"{total_failure_cases}"
)

print(
    f"Risk-group accuracy: "
    f"{risk_group_accuracy:.4%}"
)

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


# ============================================================
# 23. Display incorrect predictions in the output
# ============================================================

print()
print("=" * 70)
print("INCORRECT PREDICTIONS TO REVIEW")
print("=" * 70)

if total_failure_cases == 0:

    print()
    print(
        "No incorrect predictions were found in this risk group."
    )

else:

    display_columns = [
        column
        for column in [
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
        if column in failure_cases_dataframe.columns
    ]

    print()
    print(
        failure_cases_dataframe[
            display_columns
        ].to_string(index=False)
    )


# ============================================================
# 24. Create a contact sheet for incorrect predictions
# ============================================================

if total_failure_cases > 0:

    print()
    print("Creating failure-case contact sheet...")

    number_of_columns = min(
        4,
        total_failure_cases
    )

    number_of_rows = math.ceil(
        total_failure_cases
        / number_of_columns
    )

    figure, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=number_of_columns,
        figsize=(
            5 * number_of_columns,
            6 * number_of_rows
        )
    )

    # Convert axes into a flat list for one or multiple images.
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for image_index_number, (_, row) in enumerate(
        failure_cases_dataframe.iterrows()
    ):

        axis = axes[image_index_number]

        image_path = Path(
            row["resolved_path"]
        )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            image = ImageOps.contain(
                image,
                size=(900, 900)
            )

            axis.imshow(image)

        except Exception as error:

            axis.text(
                0.5,
                0.5,
                "Could not open image\n"
                f"{error}",
                ha="center",
                va="center",
                fontsize=10
            )

        axis.axis("off")

        image_name = row.get(
            "img_name",
            image_path.name
        )

        true_label = row.get(
            "true_label",
            "unknown"
        )

        predicted_label = row.get(
            "predicted_label",
            "unknown"
        )

        confidence = float(
            row["calibrated_confidence"]
        )

        title = (
            f"{image_name}\n"
            f"True: {true_label}\n"
            f"Predicted: {predicted_label}\n"
            f"Confidence: {confidence:.4f}\n"
            f"NCR: high"
        )

        axis.set_title(
            title,
            fontsize=10,
            color="red"
        )

    # Hide unused axes.
    for unused_axis in axes[total_failure_cases:]:
        unused_axis.axis("off")

    figure.suptitle(
        "Failure Cases: Monocyte with High "
        "Nuclear-Cytoplasmic Ratio",
        fontsize=16
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )

    figure.savefig(
        CONTACT_SHEET_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print()
    print(
        f"Failure contact sheet saved to:\n"
        f"{CONTACT_SHEET_PATH}"
    )


# ============================================================
# 25. Final message
# ============================================================

print()
print("=" * 70)
print("FAILURE-CASE EXTRACTION FINISHED")
print("=" * 70)

print()
print(
    "You can now open the failure CSV and manually review:"
)

print()
print(
    "1. Whether the image is visually ambiguous."
)

print(
    "2. Whether the nuclear-cytoplasmic ratio appears high."
)

print(
    "3. Which morphological attribute may have confused the model."
)

print(
    "4. Whether the prediction is a genuine model failure "
    "or a questionable annotation."
)
