# -*- coding: utf-8 -*-
"""Continue the v12 TSV Connection2 model with an S-parameter objective only.

Run this file directly in VS Code after `train_tsv_connection2_sparam_model.py`.
No command-line arguments are required.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skrf as rf
import torch
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_tsv_connection2_sparam_model as base


PROJECT_ROOT = THIS_DIR.parents[2]
BASE_OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v12_hfss_v08_multihead_chain" / "results" / "tsv_connection2_sparam_model"
BASE_CHECKPOINT = BASE_OUTPUT_DIR / "tsv_connection2_sparam_net.pt"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "tsv_connection2_sparam_continue"
)

RANDOM_SEED = 20260713
CONTINUE_EPOCHS = 800
PATIENCE = 160
BATCH_SIZE = 64
LR = 5e-6
WEIGHT_DECAY = 1e-8
PRINT_EVERY = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sparam_only(model, df, arrays, device):
    x_norm, _, _, train_mask, val_mask, s_target, freq, metadata = arrays
    y_mean = torch.tensor(metadata["y_log_mean"], dtype=base.REAL_DTYPE, device=device)
    y_std = torch.tensor(metadata["y_log_std"], dtype=base.REAL_DTYPE, device=device)

    train_ds = TensorDataset(
        torch.tensor(x_norm[train_mask], dtype=base.REAL_DTYPE),
        torch.tensor(df.loc[train_mask, "h_tsv"].to_numpy(dtype=np.float64), dtype=base.REAL_DTYPE),
        torch.tensor(s_target[train_mask], dtype=base.COMPLEX_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    x_val = torch.tensor(x_norm[val_mask], dtype=base.REAL_DTYPE, device=device)
    length_val = torch.tensor(df.loc[val_mask, "h_tsv"].to_numpy(dtype=np.float64), dtype=base.REAL_DTYPE, device=device)
    s_val = torch.tensor(s_target[val_mask], dtype=base.COMPLEX_DTYPE, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=45, factor=0.5)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    rows = []

    for epoch in range(1, CONTINUE_EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, lb, sb in loader:
            xb, lb, sb = xb.to(device), lb.to(device), sb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(xb)
            params = torch.exp(torch.clamp(pred_norm * y_std + y_mean, min=-40.0, max=40.0))
            pred_s = base.circuit_params_to_s_torch(params, lb, freq)
            loss = base.s_loss(pred_s, sb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)

        model.eval()
        with torch.no_grad():
            val_norm = model(x_val)
            val_params = torch.exp(torch.clamp(val_norm * y_std + y_mean, min=-40.0, max=40.0))
            val_pred_s = base.circuit_params_to_s_torch(val_params, length_val, freq)
            val_s_loss = base.s_loss(val_pred_s, s_val).item()
        train_s_loss = total / max(seen, 1)
        scheduler.step(val_s_loss)
        rows.append(
            {
                "stage": "sparam_continue",
                "epoch": epoch,
                "train_s_loss": train_s_loss,
                "val_s_loss": val_s_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_s_loss < best_val:
            best_val = val_s_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[continue] epoch={epoch}, train_s={train_s_loss:.6e}, val_s={val_s_loss:.6e}", flush=True)
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def build_comparison(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["train", "val"]:
        old = before[before["split"].eq(split)].iloc[0]
        new = after[after["split"].eq(split)].iloc[0]
        row = {"split": split}
        for col in [
            "s_mse_mean",
            "s_mse_median",
            "s11_db_mae_mean",
            "s21_db_mae_mean",
            "s11_phase_mae_deg_mean",
            "s21_phase_mae_deg_mean",
        ]:
            row[f"before_{col}"] = float(old[col])
            row[f"after_{col}"] = float(new[col])
            row[f"improvement_pct_{col}"] = float((old[col] - new[col]) / old[col] * 100.0) if float(old[col]) != 0.0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing base checkpoint: {BASE_CHECKPOINT}")
    if not base.PARAM_CSV.exists():
        raise FileNotFoundError(f"Missing extracted TSV parameter CSV: {base.PARAM_CSV}")

    df = pd.read_csv(base.PARAM_CSV, encoding="utf-8-sig").sort_values("dut_index").reset_index(drop=True)
    arrays = base.prepare_arrays(df)
    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"device={device}, samples={len(df)}", flush=True)

    checkpoint = torch.load(BASE_CHECKPOINT, map_location=device)
    model = base.TsvParamNet().to(dtype=base.REAL_DTYPE, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    before_metrics, _ = base.evaluate(model, df, arrays, device)
    before_summary = base.summarize(before_metrics)

    history = train_sparam_only(model, df, arrays, device)
    metrics, pred_df = base.evaluate(model, df, arrays, device)
    summary = base.summarize(metrics)

    base.OUTPUT_DIR = OUTPUT_DIR
    plot_dir = base.save_plots(metrics, df, pred_df, arrays[5], arrays[6])

    previous_summary = read_summary(BASE_OUTPUT_DIR / "tsv_connection2_sparam_summary.csv")
    comparison = build_comparison(previous_summary, summary)

    torch.save({"model_state_dict": model.state_dict(), "metadata": arrays[7]}, OUTPUT_DIR / "tsv_connection2_sparam_continue_net.pt")
    history.to_csv(OUTPUT_DIR / "tsv_connection2_sparam_continue_history.csv", index=False, encoding="utf-8-sig")
    before_summary.to_csv(OUTPUT_DIR / "tsv_connection2_before_loaded_summary.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "tsv_connection2_sparam_continue_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUTPUT_DIR / "tsv_connection2_sparam_continue_param_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "tsv_connection2_sparam_continue_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_DIR / "tsv_connection2_sparam_continue_comparison.csv", index=False, encoding="utf-8-sig")

    report = {
        "entry": Path(__file__).name,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "output_dir": str(OUTPUT_DIR),
        "param_csv": str(base.PARAM_CSV),
        "samples": int(len(df)),
        "train_count": int((metrics["split"] == "train").sum()),
        "val_count": int((metrics["split"] == "val").sum()),
        "epochs_sparam_continue": int(history["epoch"].max()),
        "baseline_summary_file": str(BASE_OUTPUT_DIR / "tsv_connection2_sparam_summary.csv"),
        "summary": summary.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
    }
    (OUTPUT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 TSV Connection2 S-Parameter Continue Training",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Base checkpoint: `{BASE_CHECKPOINT}`",
                f"- Output: `{OUTPUT_DIR}`",
                f"- Model: `{OUTPUT_DIR / 'tsv_connection2_sparam_continue_net.pt'}`",
                f"- Plots: `{plot_dir}`",
                "",
                "## After Continue Training",
                "",
                base.dataframe_to_markdown(summary),
                "",
                "## Before/After Comparison",
                "",
                base.dataframe_to_markdown(comparison),
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
