# -*- coding: utf-8 -*-
"""Continue training the LHS cascade multi-head model with S-parameter loss.

Run this file directly in VS Code. It loads the model produced by
``train_lhs_connection_multihead_sparam.py`` and continues optimization using
only the full-structure complex S-parameter target.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V09_CODE_DIR = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "code"
if str(V09_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V09_CODE_DIR))

import train_lhs_connection_multihead_sparam as base_train
import train_connection_network_params as param_train


SOURCE_MODEL_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v09_rdl_lhs_dataset_comparison"
    / "results"
    / "connection_multihead_lhs100_200_400_v09_rdl_all_param_pretrain_sparam"
)
SOURCE_CHECKPOINT = SOURCE_MODEL_DIR / "connection_param_multihead_net.pt"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v09_rdl_lhs_dataset_comparison"
    / "results"
    / "connection_multihead_lhs100_200_400_v09_rdl_all_sparam_continue"
)

# Continue-training controls.
RANDOM_SEED = 20260707
SPARAM_EPOCHS = 600
SPARAM_LR = 1e-5
PATIENCE_SPARAM = 120
PARAM_ANCHOR_WEIGHT = 0.0


def configure_base():
    base_train.OUTPUT_DIR = OUTPUT_DIR
    base_train.RANDOM_SEED = RANDOM_SEED
    base_train.RUN_PARAM_PRETRAIN = False
    base_train.SPARAM_EPOCHS = SPARAM_EPOCHS
    base_train.SPARAM_LR = SPARAM_LR
    base_train.PATIENCE_SPARAM = PATIENCE_SPARAM
    base_train.PARAM_ANCHOR_WEIGHT = PARAM_ANCHOR_WEIGHT
    base_train.apply_base_training_config()


def load_model(input_dim, device):
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {SOURCE_CHECKPOINT}")
    model = base_train.base.MultiHeadConnectionNet(
        input_dim=input_dim,
        connection_count=param_train.CONNECTION_COUNT,
        head_dim=len(param_train.SCALE_COLUMNS),
    ).to(dtype=base_train.REAL_DTYPE, device=device)
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dtype=base_train.REAL_DTYPE, device=device)
    print(f"Loaded source checkpoint: {SOURCE_CHECKPOINT}", flush=True)
    return model, checkpoint


def main():
    configure_base()
    base_train.set_seed(RANDOM_SEED)
    base_train.base.REAL_DTYPE = base_train.REAL_DTYPE
    base_train.base.COMPLEX_DTYPE = base_train.COMPLEX_DTYPE
    device = torch.device(
        "cuda" if base_train.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(f"S-parameter continue training only, anchor weight={PARAM_ANCHOR_WEIGHT}", flush=True)

    arrays = base_train.build_training_arrays()
    dut_df, x_norm, _, _, train_mask, val_mask, test_mask, metadata, _, _, _, _, _ = arrays
    model, source_checkpoint = load_model(x_norm.shape[1], device)

    sparam_history = base_train.train_sparam(model, arrays, device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sparam_history.to_csv(OUTPUT_DIR / "multihead_training_history.csv", index=False, encoding="utf-8-sig")

    metadata = dict(metadata)
    metadata.update(
        {
            "model_type": "multihead_connection_net_lhs_sparam_continue",
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "continue_sparam_only": True,
            "continue_sparam_lr": SPARAM_LR,
            "continue_sparam_epochs": SPARAM_EPOCHS,
            "param_anchor_weight": PARAM_ANCHOR_WEIGHT,
        }
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "model_type": metadata["model_type"],
            "connection_count": param_train.CONNECTION_COUNT,
            "head_dim": len(param_train.SCALE_COLUMNS),
            "param_anchor_weight": PARAM_ANCHOR_WEIGHT,
            "run_param_pretrain": False,
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_script": str(Path(__file__).resolve()),
        },
        OUTPUT_DIR / "connection_param_multihead_net.pt",
    )

    metrics_df, pred_df, pred_s_all = base_train.predict_metrics_and_params(model, arrays, device)
    metrics_df.to_csv(OUTPUT_DIR / "multihead_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUTPUT_DIR / "multihead_param_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df = base_train.summarize_metrics(metrics_df)
    summary_df.to_csv(OUTPUT_DIR / "multihead_sparam_summary.csv", index=False, encoding="utf-8-sig")
    plot_dir, n_plots = base_train.save_plots(metrics_df, pred_df, pred_s_all, arrays)

    report = {
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "output_dir": str(OUTPUT_DIR),
        "n_total": int(len(dut_df)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "device": str(device),
        "run_param_pretrain": False,
        "continue_sparam_only": True,
        "param_anchor_weight": PARAM_ANCHOR_WEIGHT,
        "sparam_epochs_completed": int(sparam_history["epoch"].iloc[-1]) if len(sparam_history) else 0,
        "best_val_s_loss": float(sparam_history["val_s_loss"].min()) if len(sparam_history) else None,
        "plot_dir": str(plot_dir),
        "n_plots": int(n_plots),
        "summary": summary_df.to_dict(orient="records"),
        "source_model_metadata": source_checkpoint.get("metadata", {}),
    }
    with open(OUTPUT_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nContinue training complete", flush=True)
    print(f"Model: {OUTPUT_DIR / 'connection_param_multihead_net.pt'}", flush=True)
    print(f"Summary: {OUTPUT_DIR / 'multihead_sparam_summary.csv'}", flush=True)
    print(f"Plots: {plot_dir}", flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
