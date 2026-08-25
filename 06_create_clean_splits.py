from pathlib import Path
import hashlib

import pandas as pd


# ============================================================
# 1. Locate the dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT_CANDIDATES = [
    Path("/content/WBCAtt"),

    Path(
        r"F:\projects\WBC Morphology Analysis using MAL-ViT"
        r"\datasets\WBCAtt"
    ),

    PROJECT_ROOT / "WBCAtt",
]


DATASET_ROOT = None

for candidate in DATASET_ROOT_CANDIDATES:

    if (
        candidate.exists()
        and (candidate / "annotations").exists()
    ):
        DATASET_ROOT = candidate
        break


if DATASET_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the WBCAtt dataset.\n"
        "Update DATASET_ROOT_CANDIDATES in this script."
    )


ANNOTATION_ROOT = DATASET_ROOT / "annotations"

CSV_PATHS = {
    "train": ANNOTATION_ROOT / "pbc_attr_v1_train.csv",
    "validation": ANNOTATION_ROOT / "pbc_attr_v1_val.csv",
    "test": ANNOTATION_ROOT / "test.csv",
}


# ============================================================
# 2. Output paths
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CLEAN_ANNOTATION_DIR = (
    OUTPUT_DIR / "clean_annotations"
)

CLEAN_ANNOTATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLEAN_CSV_PATHS = {
    "train": (
        CLEAN_ANNOTATION_DIR
        / "pbc_attr_v1_train_clean.csv"
    ),

    "validation": (
        CLEAN_ANNOTATION_DIR
        / "pbc_attr_v1_val_clean.csv"
    ),

    "test": (
        CLEAN_ANNOTATION_DIR
        / "test_clean.csv"
    ),
}

REMOVAL_LOG_PATH = (
    OUTPUT_DIR
    / "exact_duplicate_removal_log.csv"
)


# ============================================================
# 3. Dataset columns
# ============================================================

LABEL_COLUMN = "label"
IMAGE_COLUMN = "path"


# Higher priority means that the split is preserved.
# The test set must remain completely held out.
SPLIT_PRIORITY = {
    "train": 1,
    "validation": 2,
    "test": 3,
}


# ============================================================
# 4. Resolve image paths
# ============================================================

def resolve_image_path(raw_value, label):
    """
    Resolves an image path from the CSV file.
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
            DATASET_ROOT / "PBC_dataset_normal_DIB" / raw_path,
            DATASET_ROOT / "PBC_dataset_normal_DIB" / label / raw_path,
            DATASET_ROOT / "PBC_dataset_normal_DIB" / label / raw_path.name,
            DATASET_ROOT / "PBC_dataset_normal_DIB" / raw_path.name,
            DATASET_ROOT / raw_path.name,
        ]
    )

    checked_paths = set()

    for candidate in candidates:

        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        if candidate in checked_paths:
            continue

        checked_paths.add(candidate)

        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not resolve image path.\n"
        f"CSV value: {raw_value}\n"
        f"Label: {label}"
    )


# ============================================================
# 5. Calculate exact file hash
# ============================================================

def calculate_sha256(image_path):
    """
    Calculates the exact binary-content hash of an image.
    """

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as image_file:

        for chunk in iter(
            lambda: image_file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 6. Load and hash one split
# ============================================================

def load_and_hash_split(split_name, csv_path):
    """
    Loads a split and calculates the exact hash of every image.
    """

    dataframe = pd.read_csv(csv_path)

    required_columns = [
        LABEL_COLUMN,
        IMAGE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {split_name} CSV:\n"
            f"{missing_columns}"
        )

    dataframe["_split_name"] = split_name
    dataframe["_original_row"] = dataframe.index
    dataframe["_resolved_path"] = ""
    dataframe["_sha256"] = ""
    dataframe["_remove"] = False

    total_rows = len(dataframe)

    for index, row in dataframe.iterrows():

        label = str(
            row[LABEL_COLUMN]
        ).strip().lower()

        image_path = resolve_image_path(
            raw_value=row[IMAGE_COLUMN],
            label=label,
        )

        file_hash = calculate_sha256(
            image_path
        )

        dataframe.at[
            index,
            "_resolved_path"
        ] = str(image_path)

        dataframe.at[
            index,
            "_sha256"
        ] = file_hash

        print(
            f"Hashing {split_name}: "
            f"{index + 1}/{total_rows}",
            end="\r",
        )

    print()

    return dataframe


# ============================================================
# 7. Find duplicate groups and decide what to remove
# ============================================================

def identify_duplicates(all_data):
    """
    Finds exact duplicate image content across splits.

    The highest-priority split is preserved:
        test > validation > train
    """

    removal_log = []

    grouped = all_data.groupby(
        "_sha256",
        sort=False,
    )

    duplicate_group_number = 0

    for file_hash, group in grouped:

        if group["_split_name"].nunique() <= 1:
            continue

        duplicate_group_number += 1

        group = group.sort_values(
            by="_split_name",
            key=lambda column: column.map(
                SPLIT_PRIORITY
            ),
            ascending=False,
        )

        kept_index = group.index[0]
        kept_row = group.loc[kept_index]

        for current_index in group.index[1:]:

            current_row = group.loc[
                current_index
            ]

            all_data.at[
                current_index,
                "_remove"
            ] = True

            removal_log.append(
                {
                    "duplicate_group": (
                        duplicate_group_number
                    ),
                    "sha256": file_hash,
                    "kept_split": (
                        kept_row["_split_name"]
                    ),
                    "kept_original_row": (
                        kept_row["_original_row"]
                    ),
                    "kept_label": (
                        kept_row[LABEL_COLUMN]
                    ),
                    "kept_path": (
                        kept_row["_resolved_path"]
                    ),
                    "removed_split": (
                        current_row["_split_name"]
                    ),
                    "removed_original_row": (
                        current_row["_original_row"]
                    ),
                    "removed_label": (
                        current_row[LABEL_COLUMN]
                    ),
                    "removed_path": (
                        current_row["_resolved_path"]
                    ),
                    "reason": (
                        "Exact duplicate preserved "
                        "in higher-priority split"
                    ),
                }
            )

    return all_data, removal_log


# ============================================================
# 8. Save cleaned CSV files
# ============================================================

def save_clean_splits(all_data):
    """
    Saves cleaned versions of train, validation, and test.
    """

    internal_columns = [
        "_split_name",
        "_original_row",
        "_resolved_path",
        "_sha256",
        "_remove",
    ]

    for split_name, output_path in CLEAN_CSV_PATHS.items():

        split_data = all_data[
            all_data["_split_name"] == split_name
        ].copy()

        original_count = len(split_data)

        split_data = split_data[
            split_data["_remove"] == False
        ].copy()

        removed_count = (
            original_count - len(split_data)
        )

        split_data = split_data.drop(
            columns=internal_columns
        )

        split_data.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            f"{split_name.capitalize()} cleaned CSV saved:"
        )
        print(f"Path: {output_path}")
        print(f"Original rows: {original_count}")
        print(f"Removed rows: {removed_count}")
        print(f"Remaining rows: {len(split_data)}")


# ============================================================
# 9. Verify cleaned splits
# ============================================================

def verify_clean_splits(all_data):
    """
    Verifies that no exact duplicate hash remains across
    different cleaned splits.
    """

    cleaned_data = all_data[
        all_data["_remove"] == False
    ].copy()

    duplicate_hashes = []

    grouped = cleaned_data.groupby(
        "_sha256",
        sort=False,
    )

    for file_hash, group in grouped:

        if group["_split_name"].nunique() > 1:
            duplicate_hashes.append(
                file_hash
            )

    print()
    print("=" * 70)
    print("CLEANED SPLIT VERIFICATION")
    print("=" * 70)

    print(
        "Exact duplicate hashes remaining across splits: "
        f"{len(duplicate_hashes)}"
    )

    if len(duplicate_hashes) == 0:
        print(
            "Verification status: PASSED"
        )
    else:
        print(
            "Verification status: FAILED"
        )
        print(
            "Some exact duplicates still remain."
        )


# ============================================================
# 10. Main program
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CREATE CLEAN DATASET SPLITS")
    print("=" * 70)

    print()
    print(
        f"Dataset root:\n{DATASET_ROOT}"
    )

    split_dataframes = []

    for split_name, csv_path in CSV_PATHS.items():

        print()
        print(
            f"Loading and hashing {split_name}"
        )

        dataframe = load_and_hash_split(
            split_name=split_name,
            csv_path=csv_path,
        )

        split_dataframes.append(
            dataframe
        )

    all_data = pd.concat(
        split_dataframes,
        ignore_index=True,
    )

    print()
    print("=" * 70)
    print("IDENTIFYING EXACT DUPLICATES")
    print("=" * 70)

    all_data, removal_log = identify_duplicates(
        all_data
    )

    removal_log_dataframe = pd.DataFrame(
        removal_log
    )

    removal_log_dataframe.to_csv(
        REMOVAL_LOG_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Exact duplicate rows removed: "
        f"{len(removal_log)}"
    )

    print(
        f"Removal log saved to:\n"
        f"{REMOVAL_LOG_PATH}"
    )

    save_clean_splits(
        all_data
    )

    verify_clean_splits(
        all_data
    )

    print()
    print("=" * 70)
    print("CLEAN DATASET CREATION FINISHED")
    print("=" * 70)

    print()
    print(
        "Use the clean CSV files for the next training run."
    )
