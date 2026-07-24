# -*- coding: utf-8 -*-
"""Continue the all-150 v12 cascade model for a second S-parameter-only round.

Run this file directly in VS Code. No command-line arguments are required.

This entry loads the first all-150 continuation checkpoint, restores the
checkpoint normalization, forces all original LHS150_50_Connection2 train
samples into the training split, and continues training with only the
S-parameter objective. The original 50 test samples are kept for evaluation.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_v12_hfss_v08_symmetric_multihead as v12


BASE_RUN_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue"
)
BASE_CHECKPOINT = BASE_RUN_DIR / "v08_connection_multihead_all150_sparam_continue.pt"
SHARED_TARGETS_CSV = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "hfss_v08_symmetric_multihead_lhs150_50_connection2_new_rdl_tsv"
    / "v08_shared_optimized_targets.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round2"
)

RANDOM_SEED = 20260716
EPOCHS = 500
BATCH_SIZE = 32
LR = 2e-6
WEIGHT_DECAY = 1e-8
PRINT_EVERY = 25


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def force_original_train_test_split(dut_df: pd.DataFrame) -> pd.DataFrame:
    out = dut_df.copy()
    out["split"] = np.where(out["sample_id"].astype(str).str.contains("_train_"), "train", "test")
    return out


def prepare_arrays(dut_df: pd.DataFrame, sim, metadata: dict):
    x_raw = dut_df[v12.base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    x_mean = np.asarray(metadata["x_mean"], dtype=np.float64)
    x_std = np.maximum(np.asarray(metadata["x_std"], dtype=np.float64), 1e-30)
    x_norm = (x_raw - x_mean) / x_std
    masks = {name: dut_df["split"].eq(name).to_numpy() for name in ["train", "test"]}
    y_mean = np.asarray(metadata["y_mean"], dtype=np.float64)
    y_std = np.maximum(np.asarray(metadata["y_std"], dtype=np.float64), 1e-30)
    return x_norm, masks, y_mean, y_std, sim


def train_sparam_only(model, arrays, device):
    x_norm, masks, y_mean, y_std, sim = arrays
    train_idx = np.where(masks["train"])[0]
    train_ds = TensorDataset(
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(x_norm[train_idx], dtype=v12.base.REAL_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    omega = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=v12.base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=v12.base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=v12.base.REAL_DTYPE, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for idx_b, xb in loader:
            idx_np = idx_b.numpy()
            xb = xb.to(device)
            base_b = torch.tensor(sim.base_abcds[idx_np], dtype=v12.base.COMPLEX_DTYPE, device=device)
            target_b = torch.tensor(sim.target_s[idx_np], dtype=v12.base.COMPLEX_DTYPE, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(xb)
            p_flat = v12.base.denormalize_params(pred_norm, y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, v12.shared.CONNECTION_COUNT, len(v12.shared.V08_PARAM_NAMES))
            pred_s = v12.base.abcd2s_torch(v12.shared.cascade_with_v08_torch(v12.base, base_b, p_all, omega))
            loss = v12.base.s11_s21_mag_phase_loss_torch(pred_s, target_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        train_s_loss = total / max(seen, 1)
        rows.append({"stage": "all150_sparam_continue", "epoch": epoch, "train_s_loss": train_s_loss})
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[all150-sparam] epoch={epoch}, train_s={train_s_loss:.6e}", flush=True)
    return pd.DataFrame(rows)


def evaluate(model, dut_df, arrays, device):
    x_norm, masks, y_mean, y_std, sim = arrays
    # Reuse the existing evaluator by passing a tuple compatible with the shared helper.
    helper_arrays = (x_norm, None, masks, y_mean, y_std, sim)
    return v12.shared.evaluate_model(v12.base, model, dut_df, helper_arrays, device)


def paper_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics[["sample_id", "split", "direct_nmse_s11_s21_ri", "v08_nn_nmse_s11_s21_ri"]].copy()
    out["direct_nmse_percent"] = out["direct_nmse_s11_s21_ri"] * 100.0
    out["v08_nn_nmse_percent"] = out["v08_nn_nmse_s11_s21_ri"] * 100.0
    return (
        out.groupby("split", as_index=False)
        .agg(
            count=("sample_id", "count"),
            direct_nmse_mean_percent=("direct_nmse_percent", "mean"),
            direct_nmse_median_percent=("direct_nmse_percent", "median"),
            v08_nn_nmse_mean_percent=("v08_nn_nmse_percent", "mean"),
            v08_nn_nmse_median_percent=("v08_nn_nmse_percent", "median"),
        )
        .sort_values("split")
    )


def main() -> None:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing base checkpoint: {BASE_CHECKPOINT}")
    if not SHARED_TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing shared target table for plotting: {SHARED_TARGETS_CSV}")

    v12.base.RUN_LABEL = OUTPUT_DIR.name
    v12.base.OUTPUT_DIR = OUTPUT_DIR
    v12.base.SIMULATION_BACKEND = "hfss_equivalent_circuit"
    v12.base.USE_MODEL_SET_AS_VALIDATION = True
    v12.base.collect_samples = lambda: v12.shared.collect_v11_samples(v12.base)
    v12.base.load_single_device_simulation = v12.load_hfss_equivalent_simulation
    v12.base.PLOT_SPLIT = "test"
    v12.shared.MultiHeadV08ConnectionNet = v12.SymmetricV08ConnectionNet

    checkpoint = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    dut_df = force_original_train_test_split(v12.base.collect_samples())
    sim = v12.load_hfss_equivalent_simulation(
        dut_df,
        {
            "version": "v12",
            "continue_from": str(BASE_CHECKPOINT),
            "training_split": "all original 150 train samples",
            "objective": "S11/S21 magnitude plus wrapped phase",
        },
    )
    arrays = prepare_arrays(dut_df, sim, metadata)
    device = torch.device("cuda" if v12.base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")

    model = v12.SymmetricV08ConnectionNet(input_dim=len(metadata["feature_columns"])).to(dtype=v12.base.REAL_DTYPE, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    before_metrics, before_pred = evaluate(model, dut_df, arrays, device)
    before_summary = v12.shared.summarize_metrics(before_metrics)
    before_paper_summary = paper_summary(before_metrics)

    history = train_sparam_only(model, arrays, device)
    metrics, pred_params = evaluate(model, dut_df, arrays, device)
    summary = v12.shared.summarize_metrics(metrics)
    paper = paper_summary(metrics)

    shared_targets = pd.read_csv(SHARED_TARGETS_CSV, encoding="utf-8-sig")
    plot_dir, plot_files = v12.shared.save_comparison_plots(v12.base, model, dut_df, (arrays[0], None, arrays[1], arrays[2], arrays[3], sim), metrics, shared_targets, device)

    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, OUTPUT_DIR / "v08_connection_multihead_all150_sparam_continue.pt")
    history.to_csv(OUTPUT_DIR / "v08_all150_sparam_continue_history.csv", index=False, encoding="utf-8-sig")
    before_metrics.to_csv(OUTPUT_DIR / "v08_sparam_metrics_before_continue.csv", index=False, encoding="utf-8-sig")
    before_summary.to_csv(OUTPUT_DIR / "v08_sparam_summary_before_continue.csv", index=False, encoding="utf-8-sig")
    before_paper_summary.to_csv(OUTPUT_DIR / "paper_nmse_summary_before_continue.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "v08_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    pred_params.to_csv(OUTPUT_DIR / "v08_param_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "v08_sparam_summary.csv", index=False, encoding="utf-8-sig")
    paper.to_csv(OUTPUT_DIR / "paper_nmse_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "entry": Path(__file__).name,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "output_dir": str(OUTPUT_DIR),
        "train_count": int(arrays[1]["train"].sum()),
        "test_count": int(arrays[1]["test"].sum()),
        "epochs": int(history["epoch"].max()),
        "objective": "S11/S21 magnitude plus wrapped phase only",
        "before_summary": before_summary.to_dict(orient="records"),
        "after_summary": summary.to_dict(orient="records"),
        "after_paper_nmse_summary_percent": paper.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": [str(path) for path in plot_files],
    }
    (OUTPUT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 All-150 S-Parameter Continue Training Round 2",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Base checkpoint: `{BASE_CHECKPOINT}`",
                f"- Output: `{OUTPUT_DIR}`",
                "- Training set: all original `150` train samples from `LHS150_50_Connection2/train`.",
                "- Test set: original `50` test samples from `LHS150_50_Connection2/test`.",
                "- Filter: disabled for continuation.",
                "- Objective: S11/S21 magnitude plus wrapped phase only.",
                f"- Learning rate: `{LR}`",
                f"- Epochs: `{int(history['epoch'].max())}`",
                f"- Comparison plots: `{plot_dir}`",
                "",
                "## Before Continue",
                "",
                v12.base.dataframe_to_markdown(before_summary),
                "",
                "## After Continue",
                "",
                v12.base.dataframe_to_markdown(summary),
                "",
                "## Paper-Style NMSE Percent",
                "",
                dataframe_to_markdown(paper),
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(paper.to_string(index=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
