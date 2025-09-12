import wandb

# Not working -> wandb sub process cancels main wandb run & couldn't include in main run because of a global counter conflict
def log_training_to_wandb(history: list[dict], training_time: float, model_idx: int, root_dir: Path = WANDB_DIR):
    run = wandb.init(
        project="mia-shadow-attack",
        name=f"shadow_model_{model_idx}_training",
        group="shadow_models",
        job_type="training",
        dir=root_dir,
        reinit=True
    )

    for entry in history:
        run.log({
            f"shadow_model/{model_idx}/train_loss": entry["train_loss"],
            f"shadow_model/{model_idx}/train_accuracy": entry["train_accuracy"]
        }, step=entry["epoch"])

    run.log({
        f"shadow_model/{model_idx}/total_training_time (s)": training_time
    })

    run.finish()