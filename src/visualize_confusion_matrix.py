"""
Confusion Matrix Visualization Tool for SMS Spam Detection.

Can be run directly from terminal:
    python src/visualize_confusion_matrix.py
"""

import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_model import plot_and_save_confusion_matrix


def visualize_from_saved_metrics(
    metrics_path: str = "models/model_metrics.pkl",
    output_dir: str = "models"
):
    """
    Loads saved model evaluation metrics and generates visualization heatmaps
    for all evaluated models and the winning model.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_path}'. Please run training first: python src/train_model.py"
        )

    print(f"Loading metrics from: {metrics_path}")
    eval_bundle = joblib.load(metrics_path)

    all_models = eval_bundle.get("all_models_metrics", {})
    best_name = eval_bundle.get("best_model_name", "Winning Model")

    print(f"Generating confusion matrix visualisations (Best model: {best_name})...")
    saved_files = []

    for name, m in all_models.items():
        cm = m.get("confusion_matrix")
        if cm is not None:
            filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
            out_file = os.path.join(output_dir, filename)
            plot_and_save_confusion_matrix(cm, model_name=name, output_path=out_file)
            saved_files.append(out_file)
            print(f"  -> Saved {name} heatmap: {out_file}")

    # Also save standard confusion_matrix.png for the winning model
    winning_cm = eval_bundle.get("best_metrics", {}).get("confusion_matrix")
    if winning_cm is not None:
        best_out = os.path.join(output_dir, "confusion_matrix.png")
        plot_and_save_confusion_matrix(winning_cm, model_name=f"{best_name} (Best)", output_path=best_out)
        saved_files.append(best_out)
        print(f"  -> Saved Best Model heatmap: {best_out}")

    print("\nAll confusion matrix visualizations generated successfully!")
    return saved_files


if __name__ == "__main__":
    visualize_from_saved_metrics()
