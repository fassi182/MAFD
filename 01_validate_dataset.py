from pathlib import Path
import pandas as pd


# ============================================================
# 1. CHANGE ONLY THESE PATHS
# ============================================================

# This folder should directly contain the five class folders:
#
# Basophil
# Eosinophil
# Lymphocyte
# Monocyte
# Neutrophil
#
# Adjust this path if your actual structure is different.
DATASET_ROOT = Path(
    "/content/drive/MyDrive/datasets/WBCAtt/PBC_dataset_normal_DIB"
)

TRAIN_CSV = Path(
    "/content/drive/MyDrive/datasets/WBCAtt/annotations/pbc_attr_v1_train.csv"
)

VAL_CSV = Path(
    "/content/drive/MyDrive/datasets/WBCAtt/annotations/pbc_attr_v1_val.csv"
)

TEST_CSV = Path(
    "/content/drive/MyDrive/datasets/WBCAtt/annotations/test.csv"
)

# ============================================================
# 2. EXPECTED COLUMNS
# ============================================================

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

EXPECTED_COLUMNS = [
    "img_name",
    "label",
    *ATTRIBUTE_COLUMNS,
    "path",
]

EXPECTED_CLASSES = {
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil",
}


# ============================================================
# 3. LOAD CSV FILES
# ============================================================

def load_csv(csv_path, split_name):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{split_name} CSV file was not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    print(f"\n{split_name.upper()}")
    print("-" * 60)
    print("File:", csv_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


train_df = load_csv(TRAIN_CSV, "train")
val_df = load_csv(VAL_CSV, "validation")
test_df = load_csv(TEST_CSV, "test")


# ============================================================
# 4. CHECK COLUMNS
# ============================================================

def check_columns(df, split_name):
    actual_columns = set(df.columns)
    expected_columns = set(EXPECTED_COLUMNS)

    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns

    print(f"\n{split_name} column check")
    print("Missing columns:", missing_columns)
    print("Extra columns:", extra_columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing columns: {missing_columns}"
        )


check_columns(train_df, "Train")
check_columns(val_df, "Validation")
check_columns(test_df, "Test")


# ============================================================
# 5. CLEAN TEXT VALUES
# ============================================================

def clean_dataframe(df):
    df = df.copy()

    text_columns = [
        "img_name",
        "label",
        *ATTRIBUTE_COLUMNS,
        "path",
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Create a normalized label for checking only.
    df["label_normalized"] = (
        df["label"]
        .str.lower()
        .str.strip()
    )

    return df


train_df = clean_dataframe(train_df)
val_df = clean_dataframe(val_df)
test_df = clean_dataframe(test_df)


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

def check_missing_values(df, split_name):
    missing_counts = df[EXPECTED_COLUMNS].isna().sum()
    total_missing = int(missing_counts.sum())

    print(f"\n{split_name} missing-value check")
    print("Total missing values:", total_missing)

    if total_missing > 0:
        print(missing_counts[missing_counts > 0])


check_missing_values(train_df, "Train")
check_missing_values(val_df, "Validation")
check_missing_values(test_df, "Test")


# ============================================================
# 7. CHECK CLASS LABELS
# ============================================================

def check_labels(df, split_name):
    observed_classes = set(
        df["label_normalized"].dropna().unique()
    )

    unexpected_classes = observed_classes - EXPECTED_CLASSES

    print(f"\n{split_name} class labels")
    print("Observed classes:", sorted(observed_classes))
    print("Unexpected classes:", sorted(unexpected_classes))

    if unexpected_classes:
        raise ValueError(
            f"Unexpected labels found in {split_name}: "
            f"{unexpected_classes}"
        )


check_labels(train_df, "Train")
check_labels(val_df, "Validation")
check_labels(test_df, "Test")


# ============================================================
# 8. PRINT CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(df, split_name):
    counts = (
        df["label_normalized"]
        .value_counts()
        .sort_index()
    )

    print(f"\n{split_name} class distribution")
    print(counts)


print_class_distribution(train_df, "Train")
print_class_distribution(val_df, "Validation")
print_class_distribution(test_df, "Test")


# ============================================================
# 9. FIND CLASS FOLDERS
# ============================================================

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset root was not found:\n{DATASET_ROOT}"
    )

folder_map = {
    folder.name.lower(): folder
    for folder in DATASET_ROOT.iterdir()
    if folder.is_dir()
}

print("\nDetected image folders")
print("-" * 60)

for folder_name, folder_path in sorted(folder_map.items()):
    print(folder_name, "->", folder_path)

missing_class_folders = EXPECTED_CLASSES - set(folder_map.keys())

if missing_class_folders:
    raise FileNotFoundError(
        f"These class folders were not found: "
        f"{missing_class_folders}"
    )


# ============================================================
# 10. CHECK IMAGE PATHS
# ============================================================

def resolve_image_path(row):
    class_name = row["label_normalized"]
    image_name = row["img_name"]

    class_folder = folder_map[class_name]
    image_path = class_folder / image_name

    return image_path


def check_image_paths(df, split_name):
    missing_images = []
    resolved_paths = []

    for _, row in df.iterrows():
        image_path = resolve_image_path(row)
        resolved_paths.append(str(image_path))

        if not image_path.exists():
            missing_images.append({
                "split": split_name,
                "img_name": row["img_name"],
                "label": row["label"],
                "expected_path": str(image_path),
            })

    print(f"\n{split_name} image-path check")
    print("Rows checked:", len(df))
    print("Missing images:", len(missing_images))

    return missing_images, resolved_paths


train_missing, train_paths = check_image_paths(
    train_df,
    "train"
)

val_missing, val_paths = check_image_paths(
    val_df,
    "validation"
)

test_missing, test_paths = check_image_paths(
    test_df,
    "test"
)


# ============================================================
# 11. CHECK DUPLICATES WITHIN AND BETWEEN SPLITS
# ============================================================

def duplicate_names(df):
    return df[
        df["img_name"].duplicated(keep=False)
    ]["img_name"].tolist()


print("\nDuplicate check")
print("-" * 60)

for name, df in {
    "train": train_df,
    "validation": val_df,
    "test": test_df,
}.items():
    duplicates = duplicate_names(df)
    print(name, "duplicates:", len(duplicates))


train_names = set(train_df["img_name"])
val_names = set(val_df["img_name"])
test_names = set(test_df["img_name"])

train_val_overlap = train_names & val_names
train_test_overlap = train_names & test_names
val_test_overlap = val_names & test_names

print("\nCross-split overlap")
print("Train-validation overlap:", len(train_val_overlap))
print("Train-test overlap:", len(train_test_overlap))
print("Validation-test overlap:", len(val_test_overlap))


# ============================================================
# 12. CHECK ATTRIBUTE VALUES
# ============================================================

def print_attribute_values(df, split_name):
    print(f"\n{split_name} attribute values")
    print("-" * 60)

    for attribute in ATTRIBUTE_COLUMNS:
        values = sorted(
            df[attribute]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        print(f"{attribute}: {values}")


print_attribute_values(train_df, "Train")


# ============================================================
# 13. SAVE AUDIT RESULTS
# ============================================================

all_missing_images = (
    train_missing
    + val_missing
    + test_missing
)

audit_rows = [
    {
        "split": "train",
        "rows": len(train_df),
        "missing_images": len(train_missing),
        "duplicate_image_names": len(duplicate_names(train_df)),
        "missing_values": int(
            train_df[EXPECTED_COLUMNS].isna().sum().sum()
        ),
    },
    {
        "split": "validation",
        "rows": len(val_df),
        "missing_images": len(val_missing),
        "duplicate_image_names": len(duplicate_names(val_df)),
        "missing_values": int(
            val_df[EXPECTED_COLUMNS].isna().sum().sum()
        ),
    },
    {
        "split": "test",
        "rows": len(test_df),
        "missing_images": len(test_missing),
        "duplicate_image_names": len(duplicate_names(test_df)),
        "missing_values": int(
            test_df[EXPECTED_COLUMNS].isna().sum().sum()
        ),
    },
]

audit_df = pd.DataFrame(audit_rows)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

audit_df.to_csv(
    output_dir / "dataset_audit.csv",
    index=False
)

if all_missing_images:
    pd.DataFrame(all_missing_images).to_csv(
        output_dir / "missing_images.csv",
        index=False
    )

print("\n" + "=" * 60)
print("DATASET VALIDATION FINISHED")
print("=" * 60)
print(audit_df)

print("\nExpected checks:")
print("- Missing images should be 0.")
print("- Missing values should be 0.")
print("- Cross-split overlaps should be 0.")
print("- Unexpected classes should be empty.")
