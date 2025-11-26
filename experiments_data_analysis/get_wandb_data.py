from experiments_data_analysis.file_name_settings import WANDB_ENTITY, WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS, \
    METRICS_ANALYSIS, WANDB_PROJECT_FLOWER_SIDE_CLIENTS, WANDB_PROJECT_MIA_SIDE_CLIENTS, WANDB_PROJECT_MIA_MAIN, \
    WANDB_PROJECT_FLOWER_MAIN
from experiments_data_analysis.get_wandb_metrics_and_plots.get_metric_data import get_wandb_metrics_data


# --- main experiments ---
get_wandb_metrics_data(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT_FLOWER_MAIN,
        output_path=METRICS_ANALYSIS / "main_experiments" / "flower"
)
get_wandb_metrics_data(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT_MIA_MAIN,
        output_path=METRICS_ANALYSIS / "main_experiments" / "mia"
)

# --- side experiments ---
get_wandb_metrics_data(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS,
        output_path=METRICS_ANALYSIS / "side_experiments_shadow_models" / "mia"
)

get_wandb_metrics_data(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT_FLOWER_SIDE_CLIENTS,
        output_path=METRICS_ANALYSIS / "side_experiments_clients" / "flower"
)
get_wandb_metrics_data(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT_MIA_SIDE_CLIENTS,
        output_path=METRICS_ANALYSIS / "side_experiments_clients" / "mia"
)
