from experiments_data_analysis.file_name_settings import WANDB_ENTITY, WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS, \
    METRICS_ANALYSIS, WANDB_PROJECT_FLOWER_SIDE_CLIENTS, WANDB_PROJECT_MIA_SIDE_CLIENTS, WANDB_PROJECT_MIA_MAIN, \
    WANDB_PROJECT_FLOWER_MAIN
from experiments_data_analysis.get_wandb_metrics_and_plots.get_metric_data import get_wandb_metrics_and_plots
from config_plot_style import use_thesis_style

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
