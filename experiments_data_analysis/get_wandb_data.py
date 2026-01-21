from experiments_data_analysis.data_interface.column_names_merged_table import Col
from experiments_data_analysis.data_interface.wandb_flower_table import ColFlower
from experiments_data_analysis.file_name_settings import WANDB_ENTITY, WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS, \
    WANDB_PROJECT_FLOWER_SIDE_CLIENTS, WANDB_PROJECT_MIA_SIDE_CLIENTS, WANDB_PROJECT_MIA_MAIN, \
    WANDB_PROJECT_FLOWER_MAIN
from experiments_data_analysis.get_wandb_metrics_and_plots.create_wandb_flower_plots import \
    generate_flower_compare_runs_same_metric_from_csv
from experiments_data_analysis.get_wandb_metrics_and_plots.get_metric_data import get_wandb_metrics_and_plots
from config_plot_style import use_thesis_style
from experiments_data_analysis.file_name_settings import METRICS_ANALYSIS, COMPARE_FLOWER_FIGURES_DIR
import os

use_thesis_style()

# --- main experiments ---
get_wandb_metrics_and_plots(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT_FLOWER_MAIN,
    output_path=METRICS_ANALYSIS / "main_experiments" / "flower",
    flower_run=True
)
get_wandb_metrics_and_plots(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT_MIA_MAIN,
    output_path=METRICS_ANALYSIS / "main_experiments" / "mia"
)

# --- side experiments - shadow models ---
get_wandb_metrics_and_plots(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS,
    output_path=METRICS_ANALYSIS / "side_experiments_shadow_models" / "mia"
)

# --- side experiments - clients ---
get_wandb_metrics_and_plots(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT_FLOWER_SIDE_CLIENTS,
    output_path=METRICS_ANALYSIS / "side_experiments_clients" / "flower",
    flower_run=True
)
get_wandb_metrics_and_plots(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT_MIA_SIDE_CLIENTS,
    output_path=METRICS_ANALYSIS / "side_experiments_clients" / "mia"
)

# ----------------------------------
# Generate comparison flower plots
# ----------------------------------
os.makedirs(COMPARE_FLOWER_FIGURES_DIR, exist_ok=True)

metric_experiment = METRICS_ANALYSIS / "main_experiments" / "flower"
# Shorki, 1 epoche, iid, no protection
path1 = metric_experiment / "run0-simple_model-100-0.0-2025-11-26_19-56" / "wandb_history.csv"
# resnet, 1 epoche, iid, no protection
path2 = metric_experiment / "run1-complex_model-100-0.0-2025-11-26_20-32" / "wandb_history.csv"
# Shorki, 1 epoche, non-iid, no protection
path3 = metric_experiment / "run8-simple_model-100-0.5-2025-11-27_01-01" / "wandb_history.csv"
# resnet, 1 epoche, non-iid, no protection
path4 = metric_experiment / "run9-complex_model-100-0.5-2025-11-27_01-36" / "wandb_history.csv"

run_csv_paths = [
    path1,
    path3,
    path2,
    path4,
]

run_labels = ["Shokri-CNN (IID)", "Shokri-CNN (non-IID)", "ResNet18 (IID)", "ResNet18 (non-IID)"]
generate_flower_compare_runs_same_metric_from_csv(
    run_csv_paths=run_csv_paths,
    run_labels=run_labels,
    images_folder=COMPARE_FLOWER_FIGURES_DIR,
    metric_column_name=ColFlower.OVERFITTING_GAP_LOSS,
)
