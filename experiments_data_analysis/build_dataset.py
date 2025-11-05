# scripts/build_dataset.py
import json
import pandas as pd
from pathlib import Path
from typing import Tuple
from path_settings import EXPERIMENTS_ANALYSIS_DIR_DATA

DATA_ANALYSIS_DIR: Path = EXPERIMENTS_ANALYSIS_DIR_DATA / "big_run"
OUT_DIR: Path = EXPERIMENTS_ANALYSIS_DIR_DATA / "dataset_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map possible raw keys -> normalized snake_case
KEYMAP = {
    # Server test
    "Test-Loss (Server)": "test_loss_server",
    "Test-Accuracy (Server)": "test_accuracy_server",
    # Clients (aggregated) train/val
    "Trainings-Loss (Clients aggregiert)": "train_loss_clients",
    "Trainings-Accuracy (Clients aggregiert)": "train_accuracy_clients",
    "Validierungs-Loss (Clients aggregiert)": "val_loss_clients",
    "Validierungs-Accuracy (Clients aggregiert)": "val_accuracy_clients",
    # Overfitting gaps
    "Overfitting-Gap (Clients aggregiert, Loss)": "overfit_gap_clients_loss",
    "Overfitting-Gap (Clients aggregiert, Accuracy)": "overfit_gap_clients_accuracy",
    "Overfitting-Gap (Server, Loss)": "overfit_gap_server_loss",
    "Overfitting-Gap (Server, Accuracy)": "overfit_gap_server_accuracy",
}

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_metric_keys(d: dict) -> dict:
    """Return a new dict with normalized keys using KEYMAP; keep unknown keys as-is."""
    return {KEYMAP.get(k, k): v for k, v in d.items()}

def normalize_meta(meta: dict, run_dir: Path) -> dict:
    m = dict(meta)

    # Always have a run_id
    m.setdefault("run_id", m.get("run_name", run_dir.name))

    # Architecture normalization
    arch = str(m.get("architecture", "")).lower()
    if arch in {"resnet_18", "resnet-18", "resnet18"}:
        m["architecture"] = "resnet18"
    elif arch in {"shokri", "shokri-cnn", "shokri_cnn"}:
        m["architecture"] = "shokri_cnn"
    elif arch:
        m["architecture"] = arch  # keep custom architectures as-is

    # Regularization/protection flag (your metadata uses with_protection)
    if "with_protection" in m:
        m["regularization"] = "with" if bool(m["with_protection"]) else "without"
    # (optional) keep the original flag if you still want it around:
    # else: infer from weight_decay/dropout if you later add them

    # Distribution label from iid flag + alpha
    iid_flag = m.get("iid_data_distribution", None)
    alpha = m.get("dirichlet_alpha", None)
    try:
        alpha = float(alpha) if alpha is not None else None
    except Exception:
        alpha = None

    if iid_flag is True or (alpha is not None and alpha == 0.0):
        dist = "IID"
    elif alpha is not None:
        dist = "semi-non-iid" if alpha >= 0.3 else "non-iid"
    else:
        dist = "non-iid" if iid_flag is False else None
    if dist is not None:
        m["distribution"] = dist

    # Coerce common dtypes
    for k in ("local_epochs", "num_clients", "num_server_rounds", "seed"):
        if k in m:
            try: m[k] = int(m[k])
            except: pass
    for k in ("dirichlet_alpha", "weight_decay", "dropout", "lr"):
        if k in m:
            try: m[k] = float(m[k])
            except: pass

    return m

def load_one_run(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = normalize_meta(read_json(run_dir / "metadata.json"), run_dir)
    fl_path = run_dir / "federated_learning_rounds.json"
    mia_sum_path = run_dir / "mia_summary.json"
    mia_pc_path = run_dir / "mia_per_class.json"

    fl_raw = read_json(fl_path)
    # Support either {"rounds":[...]} or {"0":{...},"1":{...}}
    if isinstance(fl_raw, dict) and "rounds" in fl_raw and isinstance(fl_raw["rounds"], list):
        rounds_list = fl_raw["rounds"]
    elif isinstance(fl_raw, dict):
        rounds_list = [{"round": int(r), **metrics} for r, metrics in fl_raw.items()]
    else:
        raise ValueError(f"Unexpected FL JSON shape in {fl_path}")

    fl_rows = []
    for row in rounds_list:
        r = dict(row)
        r["round"] = int(r.get("round"))
        r = normalize_metric_keys(r)
        r.update(meta)
        fl_rows.append(r)
    fl_df = pd.DataFrame(fl_rows).sort_values("round").reset_index(drop=True)

    # last-round snapshot as “final”
    last_round = int(fl_df["round"].max())
    fl_final = (
        fl_df.loc[fl_df["round"].eq(last_round)]
             .drop(columns=["round"])
             .assign(last_round=last_round)
             .reset_index(drop=True)
    )

    # MIA summary: flatten {metric:{mean, std}} (accept stddev)
    mia_sum_raw = read_json(mia_sum_path)
    flat = {}
    for metric, stats in mia_sum_raw.items():
        mkey = metric.strip().lower().replace("-", "_")
        for stat, val in stats.items():
            skey = stat.strip().lower().replace("stddev", "std")
            flat[f"{mkey}_{skey}"] = val
    mia_df = pd.DataFrame([{**flat, **meta}])

    # MIA per-class: support {"classes":[{...}]} or {"0":{...}}
    mia_pc_raw = read_json(mia_pc_path)
    if isinstance(mia_pc_raw, dict) and "classes" in mia_pc_raw:
        pc_list = mia_pc_raw["classes"]
    elif isinstance(mia_pc_raw, dict):
        pc_list = [{"class": int(k), **v} for k, v in mia_pc_raw.items()]
    else:
        raise ValueError(f"Unexpected MIA per-class JSON in {mia_pc_path}")

    metric_map = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
                  "f1": "f1", "f1-score": "f1", "auc": "auc", "far": "far"}
    pc_rows = []
    for d in pc_list:
        row = {"class": int(d["class"])}
        for k, v in d.items():
            if k == "class":
                continue
            kk = metric_map.get(k.strip().lower(), k.strip().lower())
            # cast numbers if they come as strings
            try:
                row[kk] = float(v) if isinstance(v, str) else v
            except Exception:
                row[kk] = v
        row.update(meta)
        pc_rows.append(row)
    mia_pc_df = pd.DataFrame(pc_rows).sort_values("class").reset_index(drop=True)

    return fl_df, fl_final, mia_df, mia_pc_df

def load_all_runs(root: Path):
    fl_all, fl_final_all, mia_all, mia_pc_all = [], [], [], []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("run")):
        try:
            fl_df, fl_final, mia_df, mia_pc_df = load_one_run(run_dir)
        except Exception as e:
            print(f"[WARN] Skipping {run_dir.name}: {e}")
            continue
        fl_all.append(fl_df)
        fl_final_all.append(fl_final)
        mia_all.append(mia_df)
        mia_pc_all.append(mia_pc_df)

    if not fl_all:
        raise SystemExit(f"No valid runs found in {root}")

    return (
        pd.concat(fl_all, ignore_index=True),
        pd.concat(fl_final_all, ignore_index=True),
        pd.concat(mia_all, ignore_index=True),
        pd.concat(mia_pc_all, ignore_index=True),
    )

if __name__ == "__main__":
    fl_df, fl_final_df, mia_df, mia_pc_df = load_all_runs(DATA_ANALYSIS_DIR)

    fl_df.to_parquet(OUT_DIR / "fl_rounds.parquet", index=False)
    fl_final_df.to_parquet(OUT_DIR / "fl_final.parquet", index=False)
    mia_df.to_parquet(OUT_DIR / "mia_summary.parquet", index=False)
    mia_pc_df.to_parquet(OUT_DIR / "mia_per_class.parquet", index=False)

    print(f"Saved tidy tables to {OUT_DIR}")
