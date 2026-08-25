from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "split_leakage_audit_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ROWS_PER_SHEET = 10


# ============================================================
# Image loading
# ============================================================

def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


# ============================================================
# Create contact sheets
# ============================================================

def create_contact_sheet(
    dataframe,
    sheet_number,
):
    number_of_rows = len(dataframe)

    figure, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=2,
        figsize=(14, 3 * number_of_rows),
    )

    if number_of_rows == 1:
        axes = [axes]

    for row_index, (_, row) in enumerate(
        dataframe.iterrows()
    ):

        image_a = load_image(
            row["path_a"]
        )

        image_b = load_image(
            row["path_b"]
        )

        axis_a = axes[row_index][0]
        axis_b = axes[row_index][1]

        if image_a is not None:
            axis_a.imshow(image_a)
        else:
            axis_a.text(
                0.5,
                0.5,
                "Could not open image",
                ha="center",
                va="center",
            )

        if image_b is not None:
            axis_b.imshow(image_b)
        else:
            axis_b.text(
                0.5,
                0.5,
                "Could not open image",
                ha="center",
                va="center",
            )

        axis_a.axis("off")
        axis_b.axis("off")

        title_a = (
            f"{row['split_a']} row {row['row_a']}\n"
            f"Label: {row['label_a']}\n"
            f"{Path(row['path_a']).name}"
        )

        title_b = (
            f"{row['split_b']} row {row['row_b']}\n"
            f"Label: {row['label_b']}\n"
            f"{Path(row['path_b']).name}"
        )

        axis_a.set_title(
            title_a,
            fontsize=9,
        )

        axis_b.set_title(
            title_b,
            fontsize=9,
        )

        pair_information = (
            f"pHash distance: "
            f"{row['phash_distance']} | "
            f"Exact match: "
            f"{row['exact_file_match']} | "
            f"Same label: "
            f"{row['same_label']}"
        )

        figure.text(
            0.5,
            1.0 - (
                row_index + 0.5
            ) / number_of_rows,
            pair_information,
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )

    figure.suptitle(
        f"Split Leakage Candidates "
        f"Sheet {sheet_number}",
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    output_path = (
        OUTPUT_DIR
        / f"leakage_contact_sheet_{sheet_number:02d}.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved:\n{output_path}"
    )


# ============================================================
# Main program
# ============================================================

if __name__ == "__main__":

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results CSV not found:\n{RESULTS_PATH}"
        )

    dataframe = pd.read_csv(
        RESULTS_PATH
    )

    if len(dataframe) == 0:
        print(
            "The leakage results CSV is empty."
        )
        raise SystemExit

    dataframe["exact_file_match"] = (
        dataframe["exact_file_match"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    dataframe = dataframe.sort_values(
        by=[
            "exact_file_match",
            "phash_distance",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    number_of_sheets = math.ceil(
        len(dataframe) / ROWS_PER_SHEET
    )

    for sheet_index in range(number_of_sheets):

        start_index = (
            sheet_index * ROWS_PER_SHEET
        )

        end_index = (
            start_index + ROWS_PER_SHEET
        )

        sheet_dataframe = dataframe.iloc[
            start_index:end_index
        ]

        create_contact_sheet(
            dataframe=sheet_dataframe,
            sheet_number=sheet_index + 1,
        )

    print()
    print(
        "Contact-sheet generation finished."
    )
