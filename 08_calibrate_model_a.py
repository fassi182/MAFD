from pathlib import Path
import importlib.util
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader

from torchvision.models import resnet18

import matplotlib.pyplot as plt


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Paths
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


CLEAN_ANNOTATION_DIR = (
    MODEL_OUTPUT_DIR / "clean_annotations"
)

CLEAN_VALIDATION_CSV = (
    CLEAN_ANNOTATION_DIR
    / "pbc_attr_v1_val_clean.csv"
)

CLEAN_TEST_CSV = (
    CLEAN_ANNOTATION_DIR
    / "test_clean.csv"
)

MODEL_PATH = (
    MODEL_OUTPUT_DIR
    / "model_a_clean_resnet18_best.pt"
)

ANALYSIS_OUTPUT_DIR = (
    MODEL_OUTPUT_DIR
    / "calibrated_reliability_analysis"
)

ANALYSIS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CALIBRATION_SUMMARY_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "calibration_summary.csv"
)

ATTRIBUTE_RESULTS_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "calibrated_attribute_value_reliability.csv"
)

PREDICTIONS_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "calibrated_test_predictions.csv"
)

TEMPERATURE_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "temperature_scaling_result.json"
)

RELIABILITY_PLOT_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "overall_reliability_diagram.png"
)

ATTRIBUTE_ECE_PLOT_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "attribute_value_ece_comparison.png"
)


# ============================================================
# 3. Basic checks
# ============================================================

required_files = [
    CLEAN_VALIDATION_CSV,
    CLEAN_TEST_CSV,
    MODEL_PATH,
]

for required_file in required_files:

    if not required_file.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n"
            f"{required_file}"
        )


# ============================================================
# 4. Import the previous Dataset file
# ============================================================

def load_dataset_module():
    """
    Loads WBCAttDataset and transformations from the
    previous DataLoader script.
    """

    possible_files = [
        Path("/content/2_test_dataloader.py"),
        Path("/content/02_test_dataloader.py"),
        Path("2_test_dataloader.py"),
        Path("02_test_dataloader.py"),
    ]

    dataset_file = None

    for file_path in possible_files:

        if file_path.exists():
            dataset_file = file_path
            break

    if dataset_file is None:
        raise FileNotFoundError(
            "Could not find the DataLoader file.\n"
            "Expected one of:\n"
            f"{possible_files}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            "wbc_dataset_module",
            dataset_file,
        )
    )

    if specification is None:
        raise ImportError(
            "Could not create the Dataset module."
        )

    dataset_module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    if specification.loader is None:
        raise ImportError(
            "Could not load the Dataset module."
        )

    specification.loader.exec_module(
        dataset_module
    )

    return dataset_module


dataset_module = load_dataset_module()

WBCAttDataset = (
    dataset_module.WBCAttDataset
)

evaluation_transform = (
    dataset_module.evaluation_transform
)

CLASS_NAMES = (
    dataset_module.CLASS_NAMES
)

ATTRIBUTE_COLUMNS = (
    dataset_module.ATTRIBUTE_COLUMNS
)

ATTRIBUTE_VALUES = (
    dataset_module.ATTRIBUTE_VALUES
)


# ============================================================
# 5. Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("VALIDATION TEMPERATURE SCALING AND MORPHOLOGY ANALYSIS")
print("=" * 70)

print()
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# 6. Load clean validation and test datasets
# ============================================================

validation_dataset = WBCAttDataset(
    csv_path=CLEAN_VALIDATION_CSV,
    split="validation_calibration",
    transform=evaluation_transform,
)

test_dataset = WBCAttDataset(
    csv_path=CLEAN_TEST_CSV,
    split="test_calibration",
    transform=evaluation_transform,
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 7. Rebuild Model A and load the clean checkpoint
# ============================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=len(CLASS_NAMES),
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print()
print(
    f"Loaded model checkpoint from:\n"
    f"{MODEL_PATH}"
)

print(
    f"Best training epoch: "
    f"{checkpoint['epoch']}"
)


# ============================================================
# 8. Collect logits, labels, attributes, and paths
# ============================================================

def collect_model_outputs(
    model,
    data_loader,
    device,
):
    """
    Collects raw logits without applying softmax.
    """

    all_logits = []
    all_labels = []
    all_attributes = []
    all_paths = []

    model.eval()

    with torch.no_grad():

        for images, labels, attributes, paths in data_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            all_attributes.append(
                attributes.cpu()
            )

            all_paths.extend(
                list(paths)
            )

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_attributes, dim=0),
        all_paths,
    )


print()
print("Collecting validation logits...")

validation_logits, validation_labels, _, _ = (
    collect_model_outputs(
        model=model,
        data_loader=validation_loader,
        device=DEVICE,
    )
)

print(
    f"Validation logits shape: "
    f"{validation_logits.shape}"
)

print()
print("Collecting test logits...")

test_logits, test_labels, test_attributes, test_paths = (
    collect_model_outputs(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
    )
)

print(
    f"Test logits shape: "
    f"{test_logits.shape}"
)


# ============================================================
# 9. Fit temperature on validation data
# ============================================================

def fit_temperature(
    logits,
    labels,
):
    """
    Learns one scalar temperature using only validation data.

    Temperature greater than 1 usually reduces confidence.
    Temperature smaller than 1 usually increases confidence.
    """

    temperature_parameter = nn.Parameter(
        torch.zeros(
            1,
            dtype=torch.float32,
        )
    )

    optimizer = LBFGS(
        [temperature_parameter],
        lr=0.01,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    criterion = nn.CrossEntropyLoss()

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(
            temperature_parameter
        )

        scaled_logits = (
            logits / temperature
        )

        loss = criterion(
            scaled_logits,
            labels,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    final_temperature = torch.exp(
        temperature_parameter
    ).detach().item()

    return final_temperature


temperature = fit_temperature(
    logits=validation_logits,
    labels=validation_labels,
)

print()
print("=" * 70)
print("TEMPERATURE SCALING")
print("=" * 70)

print(
    f"Learned temperature: "
    f"{temperature:.6f}"
)


with open(
    TEMPERATURE_PATH,
    "w",
    encoding="utf-8",
) as temperature_file:

    json.dump(
        {
            "temperature": temperature,
            "fitted_using": "clean validation set",
            "applied_to": "clean test set",
        },
        temperature_file,
        indent=4,
    )


# ============================================================
# 10. Calibration functions
# ============================================================

def calculate_ece(
    confidence_values,
    correctness_values,
    number_of_bins=10,
):
    """
    Calculates Expected Calibration Error.
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


def calculate_prediction_statistics(
    logits,
    labels,
    temperature_value=1.0,
):
    """
    Calculates accuracy and confidence statistics.
    """

    probabilities = torch.softmax(
        logits / temperature_value,
        dim=1,
    ).numpy()

    labels_numpy = labels.numpy()

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = probabilities.max(
        axis=1
    )

    correctness = (
        predictions == labels_numpy
    )

    accuracy = correctness.mean()
    mean_confidence = confidence.mean()

    confidence_accuracy_gap = (
        mean_confidence
        - accuracy
    )

    ece = calculate_ece(
        confidence_values=confidence,
        correctness_values=correctness,
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "confidence": confidence,
        "correctness": correctness,
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "confidence_accuracy_gap": (
            confidence_accuracy_gap
        ),
        "expected_calibration_error": ece,
    }


# ============================================================
# 11. Raw and calibrated test statistics
# ============================================================

raw_test_statistics = (
    calculate_prediction_statistics(
        logits=test_logits,
        labels=test_labels,
        temperature_value=1.0,
    )
)

calibrated_test_statistics = (
    calculate_prediction_statistics(
        logits=test_logits,
        labels=test_labels,
        temperature_value=temperature,
    )
)

print()
print("=" * 70)
print("RAW VERSUS CALIBRATED TEST CONFIDENCE")
print("=" * 70)

print()
print("Raw confidence results:")

print(
    f"Accuracy: "
    f"{raw_test_statistics['accuracy']:.4f}"
)

print(
    f"Mean confidence: "
    f"{raw_test_statistics['mean_confidence']:.4f}"
)

print(
    f"Confidence-accuracy gap: "
    f"{raw_test_statistics['confidence_accuracy_gap']:.4f}"
)

print(
    f"ECE: "
    f"{raw_test_statistics['expected_calibration_error']:.4f}"
)

print()
print("Calibrated confidence results:")

print(
    f"Accuracy: "
    f"{calibrated_test_statistics['accuracy']:.4f}"
)

print(
    f"Mean confidence: "
    f"{calibrated_test_statistics['mean_confidence']:.4f}"
)

print(
    f"Confidence-accuracy gap: "
    f"{calibrated_test_statistics['confidence_accuracy_gap']:.4f}"
)

print(
    f"ECE: "
    f"{calibrated_test_statistics['expected_calibration_error']:.4f}"
)


# ============================================================
# 12. Bootstrap confidence intervals
# ============================================================

def bootstrap_group_statistics(
    confidence,
    correctness,
    number_of_bootstrap_samples=300,
):
    """
    Calculates 95 percent bootstrap confidence intervals
    for group accuracy, confidence gap, and ECE.
    """

    confidence = np.asarray(
        confidence,
        dtype=float,
    )

    correctness = np.asarray(
        correctness,
        dtype=bool,
    )

    sample_count = len(confidence)

    if sample_count < 2:
        return {
            "accuracy_ci_lower": np.nan,
            "accuracy_ci_upper": np.nan,
            "gap_ci_lower": np.nan,
            "gap_ci_upper": np.nan,
            "ece_ci_lower": np.nan,
            "ece_ci_upper": np.nan,
        }

    rng = np.random.default_rng(
        SEED
    )

    bootstrap_accuracy = []
    bootstrap_gap = []
    bootstrap_ece = []

    for _ in range(
        number_of_bootstrap_samples
    ):

        sample_indices = rng.integers(
            low=0,
            high=sample_count,
            size=sample_count,
        )

        sample_confidence = confidence[
            sample_indices
        ]

        sample_correctness = correctness[
            sample_indices
        ]

        sample_accuracy = (
            sample_correctness.mean()
        )

        sample_gap = (
            sample_confidence.mean()
            - sample_accuracy
        )

        sample_ece = calculate_ece(
            confidence_values=sample_confidence,
            correctness_values=sample_correctness,
        )

        bootstrap_accuracy.append(
            sample_accuracy
        )

        bootstrap_gap.append(
            sample_gap
        )

        bootstrap_ece.append(
            sample_ece
        )

    return {
        "accuracy_ci_lower": np.percentile(
            bootstrap_accuracy,
            2.5,
        ),

        "accuracy_ci_upper": np.percentile(
            bootstrap_accuracy,
            97.5,
        ),

        "gap_ci_lower": np.percentile(
            bootstrap_gap,
            2.5,
        ),

        "gap_ci_upper": np.percentile(
            bootstrap_gap,
            97.5,
        ),

        "ece_ci_lower": np.percentile(
            bootstrap_ece,
            2.5,
        ),

        "ece_ci_upper": np.percentile(
            bootstrap_ece,
            97.5,
        ),
    }


# ============================================================
# 13. Morphological attribute-value analysis
# ============================================================

attribute_results = []

test_attributes_numpy = (
    test_attributes.numpy()
)

for attribute_index, attribute_name in enumerate(
    ATTRIBUTE_COLUMNS
):

    possible_values = ATTRIBUTE_VALUES[
        attribute_name
    ]

    for value_index, attribute_value in enumerate(
        possible_values
    ):

        group_mask = (
            test_attributes_numpy[
                :,
                attribute_index
            ]
            == value_index
        )

        if group_mask.sum() == 0:
            continue

        raw_confidence = (
            raw_test_statistics[
                "confidence"
            ][group_mask]
        )

        raw_correctness = (
            raw_test_statistics[
                "correctness"
            ][group_mask]
        )

        calibrated_confidence = (
            calibrated_test_statistics[
                "confidence"
            ][group_mask]
        )

        calibrated_correctness = (
            calibrated_test_statistics[
                "correctness"
            ][group_mask]
        )

        raw_accuracy = (
            raw_correctness.mean()
        )

        raw_mean_confidence = (
            raw_confidence.mean()
        )

        raw_gap = (
            raw_mean_confidence
            - raw_accuracy
        )

        raw_ece = calculate_ece(
            confidence_values=raw_confidence,
            correctness_values=raw_correctness,
        )

        calibrated_accuracy = (
            calibrated_correctness.mean()
        )

        calibrated_mean_confidence = (
            calibrated_confidence.mean()
        )

        calibrated_gap = (
            calibrated_mean_confidence
            - calibrated_accuracy
        )

        calibrated_ece = calculate_ece(
            confidence_values=calibrated_confidence,
            correctness_values=calibrated_correctness,
        )

        bootstrap_results = (
            bootstrap_group_statistics(
                confidence=calibrated_confidence,
                correctness=calibrated_correctness,
                number_of_bootstrap_samples=300,
            )
        )

        attribute_results.append(
            {
                "attribute": attribute_name,
                "attribute_value": attribute_value,
                "sample_count": int(
                    group_mask.sum()
                ),

                "raw_accuracy": raw_accuracy,
                "raw_mean_confidence": (
                    raw_mean_confidence
                ),
                "raw_confidence_accuracy_gap": (
                    raw_gap
                ),
                "raw_expected_calibration_error": (
                    raw_ece
                ),

                "calibrated_accuracy": (
                    calibrated_accuracy
                ),
                "calibrated_mean_confidence": (
                    calibrated_mean_confidence
                ),
                "calibrated_confidence_accuracy_gap": (
                    calibrated_gap
                ),
                "calibrated_expected_calibration_error": (
                    calibrated_ece
                ),

                "calibrated_accuracy_ci_lower": (
                    bootstrap_results[
                        "accuracy_ci_lower"
                    ]
                ),
                "calibrated_accuracy_ci_upper": (
                    bootstrap_results[
                        "accuracy_ci_upper"
                    ]
                ),

                "calibrated_gap_ci_lower": (
                    bootstrap_results[
                        "gap_ci_lower"
                    ]
                ),
                "calibrated_gap_ci_upper": (
                    bootstrap_results[
                        "gap_ci_upper"
                    ]
                ),

                "calibrated_ece_ci_lower": (
                    bootstrap_results[
                        "ece_ci_lower"
                    ]
                ),
                "calibrated_ece_ci_upper": (
                    bootstrap_results[
                        "ece_ci_upper"
                    ]
                ),
            }
        )


attribute_results_dataframe = pd.DataFrame(
    attribute_results
)

attribute_results_dataframe = (
    attribute_results_dataframe.sort_values(
        by=[
            "calibrated_expected_calibration_error",
            "calibrated_confidence_accuracy_gap",
        ],
        ascending=[
            False,
            False,
        ],
    )
)

attribute_results_dataframe.to_csv(
    ATTRIBUTE_RESULTS_PATH,
    index=False,
)


# ============================================================
# 14. Save overall calibration summary
# ============================================================

calibration_summary = pd.DataFrame(
    [
        {
            "condition": "raw",
            "temperature": 1.0,
            "accuracy": (
                raw_test_statistics[
                    "accuracy"
                ]
            ),
            "mean_confidence": (
                raw_test_statistics[
                    "mean_confidence"
                ]
            ),
            "confidence_accuracy_gap": (
                raw_test_statistics[
                    "confidence_accuracy_gap"
                ]
            ),
            "expected_calibration_error": (
                raw_test_statistics[
                    "expected_calibration_error"
                ]
            ),
        },

        {
            "condition": "calibrated",
            "temperature": temperature,
            "accuracy": (
                calibrated_test_statistics[
                    "accuracy"
                ]
            ),
            "mean_confidence": (
                calibrated_test_statistics[
                    "mean_confidence"
                ]
            ),
            "confidence_accuracy_gap": (
                calibrated_test_statistics[
                    "confidence_accuracy_gap"
                ]
            ),
            "expected_calibration_error": (
                calibrated_test_statistics[
                    "expected_calibration_error"
                ]
            ),
        },
    ]
)

calibration_summary.to_csv(
    CALIBRATION_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 15. Save calibrated test predictions
# ============================================================

calibrated_probabilities = (
    calibrated_test_statistics[
        "probabilities"
    ]
)

calibrated_predictions = (
    calibrated_test_statistics[
        "predictions"
    ]
)

calibrated_confidence = (
    calibrated_test_statistics[
        "confidence"
    ]
)

calibrated_correctness = (
    calibrated_test_statistics[
        "correctness"
    ]
)

prediction_dataframe = pd.DataFrame(
    {
        "image_path": test_paths,

        "true_label_index": (
            test_labels.numpy()
        ),

        "true_label": [
            CLASS_NAMES[index]
            for index in test_labels.numpy()
        ],

        "predicted_label_index": (
            calibrated_predictions
        ),

        "predicted_label": [
            CLASS_NAMES[index]
            for index in calibrated_predictions
        ],

        "raw_confidence": (
            raw_test_statistics[
                "confidence"
            ]
        ),

        "calibrated_confidence": (
            calibrated_confidence
        ),

        "correct": calibrated_correctness,
    }
)

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    prediction_dataframe[
        f"raw_probability_{class_name}"
    ] = raw_test_statistics[
        "probabilities"
    ][:, class_index]

    prediction_dataframe[
        f"calibrated_probability_{class_name}"
    ] = calibrated_probabilities[
        :,
        class_index
    ]

prediction_dataframe.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 16. Reliability diagram
# ============================================================

def reliability_curve(
    confidence,
    correctness,
    number_of_bins=10,
):
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

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

        bin_centers.append(
            (lower_bound + upper_bound)
            / 2
        )

        bin_accuracies.append(
            correctness[in_bin].mean()
        )

        bin_confidences.append(
            confidence[in_bin].mean()
        )

        bin_counts.append(
            int(in_bin.sum())
        )

    return (
        bin_centers,
        bin_accuracies,
        bin_confidences,
        bin_counts,
    )


raw_curve = reliability_curve(
    confidence=raw_test_statistics[
        "confidence"
    ],
    correctness=raw_test_statistics[
        "correctness"
    ],
)

calibrated_curve = reliability_curve(
    confidence=calibrated_test_statistics[
        "confidence"
    ],
    correctness=calibrated_test_statistics[
        "correctness"
    ],
)

plt.figure(
    figsize=(8, 7)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Perfect calibration",
)

plt.plot(
    raw_curve[2],
    raw_curve[1],
    marker="o",
    label="Raw confidence",
)

plt.plot(
    calibrated_curve[2],
    calibrated_curve[1],
    marker="o",
    label="Calibrated confidence",
)

plt.xlabel(
    "Mean confidence"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Model A Reliability Diagram"
)

plt.xlim(
    0,
    1.01,
)

plt.ylim(
    0,
    1.01,
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RELIABILITY_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# 17. Attribute ECE comparison plot
# ============================================================

plot_dataframe = (
    attribute_results_dataframe.copy()
)

plot_dataframe["group_name"] = (
    plot_dataframe["attribute"]
    + "="
    + plot_dataframe["attribute_value"]
)

plot_dataframe = plot_dataframe.sort_values(
    by="calibrated_expected_calibration_error",
    ascending=True,
)

plt.figure(
    figsize=(12, 12)
)

plt.barh(
    plot_dataframe["group_name"],
    plot_dataframe[
        "raw_expected_calibration_error"
    ],
    alpha=0.6,
    label="Raw ECE",
)

plt.barh(
    plot_dataframe["group_name"],
    plot_dataframe[
        "calibrated_expected_calibration_error"
    ],
    alpha=0.8,
    label="Calibrated ECE",
)

plt.xlabel(
    "Expected calibration error"
)

plt.ylabel(
    "Morphological attribute value"
)

plt.title(
    "Calibration Error by Morphological Attribute Value"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ATTRIBUTE_ECE_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# 18. Print important findings
# ============================================================

print()
print("=" * 70)
print("HIGHEST CALIBRATED MORPHOLOGY-SPECIFIC ECE")
print("=" * 70)

print(
    attribute_results_dataframe[
        [
            "attribute",
            "attribute_value",
            "sample_count",
            "calibrated_accuracy",
            "calibrated_mean_confidence",
            "calibrated_confidence_accuracy_gap",
            "calibrated_expected_calibration_error",
            "calibrated_ece_ci_lower",
            "calibrated_ece_ci_upper",
        ]
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# 19. Final output
# ============================================================

print()
print("=" * 70)
print("CALIBRATED RELIABILITY ANALYSIS FINISHED")
print("=" * 70)

print()
print(
    f"Calibration summary:\n"
    f"{CALIBRATION_SUMMARY_PATH}"
)

print()
print(
    f"Attribute-value results:\n"
    f"{ATTRIBUTE_RESULTS_PATH}"
)

print()
print(
    f"Calibrated predictions:\n"
    f"{PREDICTIONS_PATH}"
)

print()
print(
    f"Temperature result:\n"
    f"{TEMPERATURE_PATH}"
)

print()
print(
    f"Reliability diagram:\n"
    f"{RELIABILITY_PLOT_PATH}"
)

print()
print(
    f"Attribute ECE plot:\n"
    f"{ATTRIBUTE_ECE_PLOT_PATH}"
)
