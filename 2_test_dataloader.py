from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# 1. Dataset paths
# ============================================================
DATASET_ROOT = Path(
    "/content/WBCAtt"
)
IMAGE_ROOT = DATASET_ROOT / "PBC_dataset_normal_DIB"
ANNOTATION_ROOT = DATASET_ROOT / "annotations"

CSV_PATHS = {
    "train": ANNOTATION_ROOT / "pbc_attr_v1_train.csv",
    "validation": ANNOTATION_ROOT / "pbc_attr_v1_val.csv",
    "test": ANNOTATION_ROOT / "test.csv",
}


# ============================================================
# 2. Dataset definitions
# ============================================================

LABEL_COLUMN_CANDIDATES = [
    "label",
    "label_normalized",
    "class",
    "cell_type",
    "cell_class",
    "category",
]

IMAGE_COLUMN_CANDIDATES = [
    "image_path",
    "img_path",
    "file_path",
    "path",
    "filename",
    "file_name",
    "image",
    "image_name",
    "image_id",
]


CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


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


ATTRIBUTE_TO_INDEX = {
    attribute_name: {
        value: index
        for index, value in enumerate(values)
    }
    for attribute_name, values in ATTRIBUTE_VALUES.items()
}


# ============================================================
# 3. Basic validation functions
# ============================================================

def normalize_text(value):
    """
    Converts a CSV value into normalized lowercase text.
    """

    return str(value).strip().lower()


def infer_label_column(dataframe):
    """
    Finds the actual label column in the CSV file.

    The validation script may have created a temporary
    label_normalized Series, but that does not mean the CSV
    contains a column with that name.
    """

    expected_classes = set(CLASS_NAMES)

    # First inspect common label-column names.
    for column in LABEL_COLUMN_CANDIDATES:

        if column not in dataframe.columns:
            continue

        values = (
            dataframe[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        observed_classes = set(values.unique())

        if (
            len(observed_classes) > 0
            and observed_classes.issubset(expected_classes)
            and len(observed_classes.intersection(expected_classes)) >= 2
        ):
            return column

    # If no common name works, inspect every column.
    for column in dataframe.columns:

        values = (
            dataframe[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if len(values) == 0:
            continue

        observed_classes = set(values.unique())

        if (
            observed_classes.issubset(expected_classes)
            and len(observed_classes.intersection(expected_classes)) >= 2
        ):
            return column

    raise ValueError(
        "Could not identify the label column.\n"
        f"Available CSV columns: {list(dataframe.columns)}"
    )


def resolve_image_path(raw_value, label):
    """
    Resolves the image path from different possible CSV formats.

    Supported formats include:

    1. Absolute path
    2. Filename only
    3. Path relative to the dataset folder
    4. Path relative to the image root
    5. Path relative to the class folder
    """

    value = str(raw_value).strip()
    raw_path = Path(value)

    candidates = []

    # Case 1: absolute path
    if raw_path.is_absolute():
        candidates.append(raw_path)

    # Other possible path formats
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
        f"Could not resolve image path.\n"
        f"CSV value: {value}\n"
        f"Class label: {label}\n"
        f"Image root: {IMAGE_ROOT}"
    )


def can_resolve_image_column(dataframe, column, label_column):
    """
    Tests whether a possible column contains valid image paths.
    """

    sample_rows = dataframe.head(25)
    successful_paths = 0

    for _, row in sample_rows.iterrows():

        label = normalize_text(row[label_column])

        try:
            resolve_image_path(
                raw_value=row[column],
                label=label,
            )

            successful_paths += 1

        except FileNotFoundError:
            pass

    required_successes = min(5, len(sample_rows))

    return successful_paths >= required_successes


def infer_image_column(dataframe, label_column):
    """
    Finds the image-path column automatically.
    """

    # First inspect common image-column names.
    for column in IMAGE_COLUMN_CANDIDATES:

        if column not in dataframe.columns:
            continue

        if can_resolve_image_column(
            dataframe=dataframe,
            column=column,
            label_column=label_column,
        ):
            return column

    # If no common name works, inspect remaining columns.
    excluded_columns = set(
        [label_column] + ATTRIBUTE_COLUMNS
    )

    possible_columns = [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]

    for column in possible_columns:

        if can_resolve_image_column(
            dataframe=dataframe,
            column=column,
            label_column=label_column,
        ):
            return column

    raise ValueError(
        "Could not identify the image-path column.\n"
        f"Available CSV columns: {list(dataframe.columns)}"
    )


# ============================================================
# 4. Image transformations
# ============================================================

IMAGE_SIZE = 224

train_transform = transforms.Compose(
    [
        transforms.Resize(
            size=(IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


evaluation_transform = transforms.Compose(
    [
        transforms.Resize(
            size=(IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# 5. WBCAtt PyTorch Dataset
# ============================================================

class WBCAttDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file does not exist:\n{self.csv_path}"
            )

        self.dataframe = pd.read_csv(self.csv_path)

        print()
        print("-" * 60)
        print(f"Preparing {self.split} dataset")
        print(f"CSV file: {self.csv_path}")
        print(f"CSV columns: {list(self.dataframe.columns)}")

        # Automatically find the label column.
        self.label_column = infer_label_column(
            self.dataframe
        )

        print(
            f"Detected label column: "
            f"{self.label_column}"
        )

        # Check all attribute columns.
        missing_attributes = [
            column
            for column in ATTRIBUTE_COLUMNS
            if column not in self.dataframe.columns
        ]

        if missing_attributes:
            raise ValueError(
                "The following attribute columns are missing:\n"
                f"{missing_attributes}"
            )

        # Automatically find the image column.
        self.image_column = infer_image_column(
            dataframe=self.dataframe,
            label_column=self.label_column,
        )

        print(
            f"Detected image column: "
            f"{self.image_column}"
        )

        self.image_paths = []
        self.labels = []
        self.attributes = []

        # Process every row.
        for row_number, row in self.dataframe.iterrows():

            # Read and normalize the class label.
            label = normalize_text(
                row[self.label_column]
            )

            if label not in CLASS_TO_INDEX:
                raise ValueError(
                    f"Unexpected class label at row "
                    f"{row_number}: {label}\n"
                    f"Expected classes: {CLASS_NAMES}"
                )

            # Resolve image path.
            image_path = resolve_image_path(
                raw_value=row[self.image_column],
                label=label,
            )

            # Encode the 11 morphological attributes.
            attribute_vector = []

            for attribute_name in ATTRIBUTE_COLUMNS:

                value = normalize_text(
                    row[attribute_name]
                )

                valid_values = ATTRIBUTE_TO_INDEX[
                    attribute_name
                ]

                if value not in valid_values:
                    raise ValueError(
                        f"Unexpected attribute value.\n"
                        f"Row: {row_number}\n"
                        f"Attribute: {attribute_name}\n"
                        f"Observed value: {value}\n"
                        f"Expected values: "
                        f"{list(valid_values.keys())}"
                    )

                encoded_value = valid_values[value]
                attribute_vector.append(encoded_value)

            self.image_paths.append(image_path)

            self.labels.append(
                CLASS_TO_INDEX[label]
            )

            self.attributes.append(
                attribute_vector
            )

        # Convert labels and attributes to tensors.
        self.labels = torch.tensor(
            self.labels,
            dtype=torch.long,
        )

        self.attributes = torch.tensor(
            self.attributes,
            dtype=torch.long,
        )

        print(
            f"{self.split.capitalize()} dataset loaded successfully"
        )
        print(f"Number of samples: {len(self.image_paths)}")
        print(
            f"Attribute tensor shape: "
            f"{self.attributes.shape}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as error:
            raise RuntimeError(
                f"Could not open image:\n{image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]
        attribute_vector = self.attributes[index]

        return (
            image,
            label,
            attribute_vector,
            str(image_path),
        )


# ============================================================
# 6. DataLoader creation
# ============================================================

def create_dataloader(
    dataset,
    batch_size,
    shuffle,
):
    """
    Creates a Windows-compatible DataLoader.

    num_workers=0 is used initially to avoid multiprocessing
    problems while testing on Windows.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# 7. Inspect one batch
# ============================================================

def inspect_loader(
    loader,
    split_name,
):
    """
    Loads and validates one batch.
    """

    images, labels, attributes, image_paths = next(
        iter(loader)
    )

    print()
    print("=" * 60)
    print(f"{split_name.upper()} DATALOADER CHECK")
    print("=" * 60)

    print(
        f"Image batch shape: "
        f"{images.shape}"
    )

    print(
        f"Label batch shape: "
        f"{labels.shape}"
    )

    print(
        f"Attribute batch shape: "
        f"{attributes.shape}"
    )

    print(
        f"Number of paths in batch: "
        f"{len(image_paths)}"
    )

    print(
        f"First label index: "
        f"{labels[0].item()}"
    )

    first_class_index = labels[0].item()

    print(
        f"First label name: "
        f"{CLASS_NAMES[first_class_index]}"
    )

    print(
        f"First attribute vector: "
        f"{attributes[0].tolist()}"
    )

    print(
        f"First image path: "
        f"{image_paths[0]}"
    )

    # Required shape checks.
    assert images.ndim == 4
    assert images.shape[0] <= BATCH_SIZE
    assert images.shape[1] == 3
    assert images.shape[2] == IMAGE_SIZE
    assert images.shape[3] == IMAGE_SIZE

    assert labels.ndim == 1
    assert labels.shape[0] == images.shape[0]

    assert attributes.ndim == 2
    assert attributes.shape[0] == images.shape[0]
    assert attributes.shape[1] == len(ATTRIBUTE_COLUMNS)

    assert len(image_paths) == images.shape[0]

    # Check that the image tensor does not contain NaN
    # or infinite values.
    assert torch.isfinite(images).all()

    # Check valid class-index range.
    assert torch.all(labels >= 0)
    assert torch.all(labels < len(CLASS_NAMES))

    print("Status: PASSED")


# ============================================================
# 8. Main program
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WBCAtt DATASET AND DATALOADER TEST")
    print("=" * 60)

    print()
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Image root: {IMAGE_ROOT}")

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image root does not exist:\n{IMAGE_ROOT}"
        )

    # Batch size is deliberately small during testing.
    BATCH_SIZE = 8

    # Create datasets.
    train_dataset = WBCAttDataset(
        csv_path=CSV_PATHS["train"],
        split="train",
        transform=train_transform,
    )

    validation_dataset = WBCAttDataset(
        csv_path=CSV_PATHS["validation"],
        split="validation",
        transform=evaluation_transform,
    )

    test_dataset = WBCAttDataset(
        csv_path=CSV_PATHS["test"],
        split="test",
        transform=evaluation_transform,
    )

    # Create DataLoaders.
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Inspect one batch from every split.
    inspect_loader(
        loader=train_loader,
        split_name="train",
    )

    inspect_loader(
        loader=validation_loader,
        split_name="validation",
    )

    inspect_loader(
        loader=test_loader,
        split_name="test",
    )

    # Display output sizes required by Model B.
    print()
    print("=" * 60)
    print("MODEL B ATTRIBUTE OUTPUT SIZES")
    print("=" * 60)

    for attribute_name in ATTRIBUTE_COLUMNS:

        number_of_classes = len(
            ATTRIBUTE_VALUES[attribute_name]
        )

        print(
            f"{attribute_name}: "
            f"{number_of_classes} possible values"
        )

    print()
    print("=" * 60)
    print("DATASET AND DATALOADER TEST FINISHED SUCCESSFULLY")
    print("=" * 60)
