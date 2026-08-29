%%writefile /content/13_gradcam_failure_cases.py

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn.functional as F


sys.path.insert(
    0,
    "/content"
)

from mafd_utils import (
    MODEL_OUTPUT_DIR,
    FAILURE_ANALYSIS_DIR,
    load_model,
    get_evaluation_transform,
    CLASS_TO_INDEX,
    clean_prediction_dataframe,
    safe_filename,
)


# ============================================================
# Paths
# ============================================================

COMPARISON_PATH = (
    FAILURE_ANALYSIS_DIR
    / "failure_control_comparison.csv"
)

GRADCAM_DIR = (
    FAILURE_ANALYSIS_DIR
    / "gradcam_outputs"
)

GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRADCAM_RESULTS_PATH = (
    FAILURE_ANALYSIS_DIR
    / "gradcam_results.csv"
)


# ============================================================
# Grad-CAM implementation
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activations
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradients
            )
        )

    def save_activations(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output.detach()

    def save_gradients(
        self,
        module,
        grad_inputs,
        grad_outputs,
    ):
        self.gradients = (
            grad_outputs[0].detach()
        )

    def generate(
        self,
        image_tensor,
        target_index,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            image_tensor
        )

        score = logits[
            0,
            target_index
        ]

        score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0
        ].cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        probabilities = torch.softmax(
            logits.detach(),
            dim=1,
        )[0].cpu().numpy()

        return cam, probabilities

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# Helper functions
# ============================================================

def create_overlay(
    image,
    cam,
):
    image_array = np.asarray(
        image
    ).astype(np.float32) / 255.0

    heatmap = plt.cm.jet(
        cam
    )[
        ...,
        :3
    ]

    overlay = (
        0.55 * image_array
        + 0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1,
    )

    return overlay


# ============================================================
# Load data and model
# ============================================================

if not COMPARISON_PATH.exists():
    raise FileNotFoundError(
        "Run 12_compare_failure_controls.py first.\n"
        f"Missing file:\n{COMPARISON_PATH}"
    )

comparison_dataframe = pd.read_csv(
    COMPARISON_PATH
)

comparison_dataframe = (
    clean_prediction_dataframe(
        comparison_dataframe
    )
)

model, device = load_model()
transform = get_evaluation_transform()

# The final convolutional layer of ResNet-18.
target_layer = model.layer4[-1].conv2

gradcam = GradCAM(
    model=model,
    target_layer=target_layer,
)


# ============================================================
# Generate Grad-CAM images
# ============================================================

results = []

for _, row in comparison_dataframe.iterrows():

    image_path = Path(
        row["resolved_path"]
    )

    image_name = row["img_name"]

    if not image_path.exists():
        print(
            f"Skipping missing image: {image_path}"
        )
        continue

    true_label = str(
        row["true_label"]
    ).strip().lower()

    predicted_label = str(
        row["predicted_label"]
    ).strip().lower()

    if predicted_label not in CLASS_TO_INDEX:
        print(
            f"Skipping unknown predicted label: "
            f"{predicted_label}"
        )
        continue

    target_index = CLASS_TO_INDEX[
        predicted_label
    ]

    image = Image.open(
        image_path
    ).convert("RGB")

    original_size = image.size

    image_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(device)

    cam, probabilities = gradcam.generate(
        image_tensor=image_tensor,
        target_index=target_index,
    )

    cam_for_original_size = torch.tensor(
        cam
    ).unsqueeze(
        0
    ).unsqueeze(
        0
    )

    cam_for_original_size = F.interpolate(
        cam_for_original_size,
        size=(
            original_size[1],
            original_size[0],
        ),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    overlay = create_overlay(
        image=image,
        cam=cam_for_original_size,
    )

    output_name = (
        safe_filename(
            Path(image_name).stem
        )
        + "_"
        + str(row["case_type"])
        + "_gradcam.png"
    )

    output_path = (
        GRADCAM_DIR
        / output_name
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
    )

    axes[0].imshow(image)
    axes[0].set_title(
        "Original image"
    )
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Attention for predicted class:\n"
        f"{predicted_label}"
    )
    axes[1].axis("off")

    figure.suptitle(
        f"{image_name}\n"
        f"True: {true_label} | "
        f"Predicted: {predicted_label} | "
        f"Confidence: "
        f"{float(row['calibrated_confidence']):.4f}",
        fontsize=11,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    results.append(
        {
            "img_name": image_name,
            "case_type": row["case_type"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "calibrated_confidence": (
                float(
                    row["calibrated_confidence"]
                )
            ),
            "gradcam_target_label": predicted_label,
            "gradcam_target_index": target_index,
            "gradcam_image_path": str(
                output_path
            ),
            "probability_basophil": (
                probabilities[
                    CLASS_TO_INDEX["basophil"]
                ]
            ),
            "probability_eosinophil": (
                probabilities[
                    CLASS_TO_INDEX["eosinophil"]
                ]
            ),
            "probability_lymphocyte": (
                probabilities[
                    CLASS_TO_INDEX["lymphocyte"]
                ]
            ),
            "probability_monocyte": (
                probabilities[
                    CLASS_TO_INDEX["monocyte"]
                ]
            ),
            "probability_neutrophil": (
                probabilities[
                    CLASS_TO_INDEX["neutrophil"]
                ]
            ),
        }
    )

gradcam.close()

results_dataframe = pd.DataFrame(
    results
)

results_dataframe.to_csv(
    GRADCAM_RESULTS_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Final output
# ============================================================

print("=" * 70)
print("GRAD-CAM ANALYSIS FINISHED")
print("=" * 70)

print()
print(
    f"Grad-CAM images saved in:\n{GRADCAM_DIR}"
)

print()
print(
    f"Grad-CAM results saved to:\n"
    f"{GRADCAM_RESULTS_PATH}"
)

print()
print(
    "For failure cases, check whether the model focused "
    "on the nucleus, cytoplasm, cell boundary, or background."
)
