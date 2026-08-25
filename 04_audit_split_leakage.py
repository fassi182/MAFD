from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from sklearn.neighbors import NearestNeighbors


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
IMAGE_ROOT = DATASET_ROOT / "PBC_dataset_normal_DIB"


CSV_PATHS = {
    "train": ANNOTATION_ROOT / "pbc_attr_v1_train.csv",
    "validation": ANNOTATION_ROOT / "pbc_attr_v1_val.csv",
    "test": ANNOTATION_ROOT / "test.csv",
}


# Save results to Google Drive when available.
DRIVE_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/MAFD_OUTPUTS"
)

if DRIVE_OUTPUT_DIR.exists():
    OUTPUT_DIR = DRIVE_OUTPUT_DIR
else:
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUTPUT_DIR / "split_leakage_audit_results.csv"
)


# ============================================================
# 2. Configuration
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

PHASH_DISTANCE_THRESHOLD = 8


# ============================================================
# 3. Path resolution
# ============================================================

def normalize_path_text(value):
    """
    Normalizes path separators so that paths written on
    Windows can also be handled in Colab.
    """

    return str(value).strip().replace("\\", "/")


def resolve_image_path(raw_value, label):
    """
    Resolves an image path from the CSV file.
    """

    value = normalize_path_text(raw_value)
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

        if candidate in checked:
            continue

        checked.add(candidate)

        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not resolve image path.\n"
        f"CSV value: {raw_value}\n"
        f"Label: {label}"
    )


# ============================================================
# 4. Load split records
# ============================================================

def load_split_records(split_name, csv_path):
    """
    Loads all image records from one split.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist:\n{csv_path}"
        )

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
            f"{missing_columns}\n"
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )

    records = []

    for row_number, row in dataframe.iterrows():

        label = str(
            row[LABEL_COLUMN]
        ).strip().lower()

        if label not in CLASS_NAMES:
            raise ValueError(
                f"Unexpected label in {split_name}, "
                f"row {row_number}: {label}"
            )

        image_path = resolve_image_path(
            raw_value=row[IMAGE_COLUMN],
            label=label,
        )

        records.append(
            {
                "split": split_name,
                "row_number": row_number,
                "label": label,
                "path": str(image_path),
            }
        )

    print(
        f"{split_name.capitalize()} records loaded: "
        f"{len(records)}"
    )

    return records


# ============================================================
# 5. Exact file hash
# ============================================================

def calculate_sha256(image_path):
    """
    Calculates the exact file-content hash.
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
# 6. Perceptual hash
# ============================================================

def calculate_perceptual_hash(image_path):
    """
    Calculates a perceptual hash using a low-frequency DCT.

    Images with a small Hamming distance are visually similar.
    """

    image = Image.open(image_path).convert("L")

    image = image.resize(
        (32, 32),
        Image.Resampling.LANCZOS,
    )

    pixels = np.asarray(
        image,
        dtype=np.float32,
    )

    dct_rows = dct(
        pixels,
        type=2,
        norm="ortho",
        axis=0,
    )

    dct_image = dct(
        dct_rows,
        type=2,
        norm="ortho",
        axis=1,
    )

    low_frequency = dct_image[:8, :8]

    # Exclude the DC coefficient from the median.
    coefficients = low_frequency.flatten()[1:]

    median_value = np.median(coefficients)

    bits = low_frequency > median_value

    return "".join(
        "1" if bit else "0"
        for bit in bits.flatten()
    )


def hash_to_uint8_vector(hash_string):
    """
    Converts a binary perceptual hash into a NumPy vector.
    """

    return np.array(
        [
            int(bit)
            for bit in hash_string
        ],
        dtype=np.uint8,
    )


# ============================================================
# 7. Compute hashes for all images
# ============================================================

def compute_hashes(records):
    """
    Computes exact and perceptual hashes for all images.
    """

    updated_records = []

    total = len(records)

    for index, record in enumerate(records, start=1):

        image_path = Path(record["path"])

        print(
            f"Hashing {record['split']}: "
            f"{index}/{total}",
            end="\r",
        )

        exact_hash = calculate_sha256(
            image_path
        )

        perceptual_hash = calculate_perceptual_hash(
            image_path
        )

        updated_record = dict(record)

        updated_record["sha256"] = exact_hash
        updated_record["phash"] = perceptual_hash

        updated_records.append(updated_record)

    print()

    return updated_records


# ============================================================
# 8. Exact cross-split duplicate detection
# ============================================================

def find_exact_cross_split_matches(
    records_a,
    records_b,
):
    """
    Finds images with identical file content across
    two different splits.
    """

    hash_to_records = {}

    for record in records_a:

        file_hash = record["sha256"]

        hash_to_records.setdefault(
            file_hash,
            [],
        ).append(record)

    matches = []

    for record_b in records_b:

        file_hash = record_b["sha256"]

        if file_hash not in hash_to_records:
            continue

        for record_a in hash_to_records[file_hash]:

            matches.append(
                {
                    "split_a": record_a["split"],
                    "row_a": record_a["row_number"],
                    "label_a": record_a["label"],
                    "path_a": record_a["path"],
                    "split_b": record_b["split"],
                    "row_b": record_b["row_number"],
                    "label_b": record_b["label"],
                    "path_b": record_b["path"],
                    "phash_distance": 0,
                    "exact_file_match": True,
                    "same_label": (
                        record_a["label"]
                        == record_b["label"]
                    ),
                }
            )

    return matches


# ============================================================
# 9. Perceptual cross-split duplicate detection
# ============================================================

def find_perceptual_cross_split_matches(
    records_a,
    records_b,
    distance_threshold,
):
    """
    Finds the nearest perceptual-hash image from split A
    for every image in split B.
    """

    hashes_a = np.stack(
        [
            hash_to_uint8_vector(
                record["phash"]
            )
            for record in records_a
        ]
    )

    hashes_b = np.stack(
        [
            hash_to_uint8_vector(
                record["phash"]
            )
            for record in records_b
        ]
    )

    nearest_model = NearestNeighbors(
        n_neighbors=1,
        metric="hamming",
        algorithm="brute",
        n_jobs=-1,
    )

    nearest_model.fit(hashes_a)

    distances, indices = nearest_model.kneighbors(
        hashes_b
    )

    matches = []

    for query_index in range(len(records_b)):

        nearest_index = indices[
            query_index,
            0,
        ]

        normalized_distance = distances[
            query_index,
            0,
        ]

        hamming_distance = int(
            round(normalized_distance * 64)
        )

        if hamming_distance > distance_threshold:
            continue

        record_a = records_a[nearest_index]
        record_b = records_b[query_index]

        matches.append(
            {
                "split_a": record_a["split"],
                "row_a": record_a["row_number"],
                "label_a": record_a["label"],
                "path_a": record_a["path"],
                "split_b": record_b["split"],
                "row_b": record_b["row_number"],
                "label_b": record_b["label"],
                "path_b": record_b["path"],
                "phash_distance": hamming_distance,
                "exact_file_match": (
                    record_a["sha256"]
                    == record_b["sha256"]
                ),
                "same_label": (
                    record_a["label"]
                    == record_b["label"]
                ),
            }
        )

    return matches


# ============================================================
# 10. Main audit
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRAIN, VALIDATION, AND TEST SPLIT LEAKAGE AUDIT")
    print("=" * 70)

    print()
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image root does not exist:\n{IMAGE_ROOT}"
        )

    # Load records.
    split_records = {}

    for split_name, csv_path in CSV_PATHS.items():

        split_records[split_name] = load_split_records(
            split_name=split_name,
            csv_path=csv_path,
        )

    # Compute hashes.
    print()
    print("=" * 70)
    print("COMPUTING IMAGE HASHES")
    print("=" * 70)

    for split_name in split_records:

        split_records[split_name] = compute_hashes(
            split_records[split_name]
        )

    # Compare every pair of splits.
    split_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    all_exact_matches = []
    all_perceptual_matches = []

    print()
    print("=" * 70)
    print("CROSS-SPLIT COMPARISON")
    print("=" * 70)

    for split_a, split_b in split_pairs:

        records_a = split_records[split_a]
        records_b = split_records[split_b]

        print()
        print(
            f"Comparing {split_a} with {split_b}"
        )

        exact_matches = find_exact_cross_split_matches(
            records_a=records_a,
            records_b=records_b,
        )

        perceptual_matches = (
            find_perceptual_cross_split_matches(
                records_a=records_a,
                records_b=records_b,
                distance_threshold=PHASH_DISTANCE_THRESHOLD,
            )
        )

        all_exact_matches.extend(
            exact_matches
        )

        all_perceptual_matches.extend(
            perceptual_matches
        )

        print(
            f"Exact file matches: "
            f"{len(exact_matches)}"
        )

        print(
            f"Perceptual matches with distance <= "
            f"{PHASH_DISTANCE_THRESHOLD}: "
            f"{len(perceptual_matches)}"
        )

    # Remove duplicate rows where an exact match is also
    # included in the perceptual matches.
    exact_keys = {
        (
            row["split_a"],
            row["row_a"],
            row["split_b"],
            row["row_b"],
        )
        for row in all_exact_matches
    }

    filtered_perceptual_matches = []

    for row in all_perceptual_matches:

        key = (
            row["split_a"],
            row["row_a"],
            row["split_b"],
            row["row_b"],
        )

        if key not in exact_keys:
            filtered_perceptual_matches.append(row)

    all_matches = (
        all_exact_matches
        + filtered_perceptual_matches
    )

    results_dataframe = pd.DataFrame(
        all_matches,
        columns=[
            "split_a",
            "row_a",
            "label_a",
            "path_a",
            "split_b",
            "row_b",
            "label_b",
            "path_b",
            "phash_distance",
            "exact_file_match",
            "same_label",
        ],
    )

    if len(results_dataframe) > 0:
        results_dataframe = results_dataframe.sort_values(
            by=[
                "exact_file_match",
                "phash_distance",
            ],
            ascending=[
                False,
                True,
            ],
        )

    results_dataframe.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # Summary.
    exact_count = len(all_exact_matches)
    perceptual_count = len(
        filtered_perceptual_matches
    )

    print()
    print("=" * 70)
    print("LEAKAGE AUDIT SUMMARY")
    print("=" * 70)

    print(
        f"Exact cross-split duplicate files: "
        f"{exact_count}"
    )

    print(
        f"Additional perceptual near-duplicates: "
        f"{perceptual_count}"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{RESULTS_PATH}"
    )

    if exact_count == 0 and perceptual_count == 0:

        print()
        print(
            "AUDIT STATUS: PASSED"
        )

        print(
            "No exact or close perceptual cross-split "
            "duplicates were detected using the configured "
            "threshold."
        )

    else:

        print()
        print(
            "AUDIT STATUS: REVIEW REQUIRED"
        )

        print(
            "Open the results CSV and manually inspect "
            "the reported image pairs before using the "
            "test score as final research evidence."
        )

    print()
    print("=" * 70)
    print("SPLIT LEAKAGE AUDIT FINISHED")
    print("=" * 70)
