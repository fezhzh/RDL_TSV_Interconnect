# -*- coding: utf-8 -*-
"""Train a TSV single-device model for LHS400_Connection2.

Run this file directly in VS Code after `extract_tsv_connection2_params.py`.
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
PARAM_CSV = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "tsv_connection2_extracted_params"
    / "TSV_connection2_circuit_params.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v12_hfss_v08_multihead_chain" / "results" / "tsv_connection2_sparam_model"

FEATURES = ["r_tsv", "h_tsv", "pitch"]
TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
RANDOM_SEED = 20260712
TRAIN_RATIO = 0.8
USE_CUDA_IF_AVAILABLE = True
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
Z_REF = 50.0

PARAM_EPOCHS = 900
PARAM_PATIENCE = 140
SPARAM_EPOCHS = 450
SPARAM_PATIENCE = 100
BATCH_SIZE = 64
PARAM_LR = 8e-4
SPARAM_LR = 2e-5
WEIGHT_DECAY = 1e-7
PARAM_ANCHOR_WEIGHT = 0.05
PRINT_EVERY = 50
PLOT_WORST_N = 10
PLOT_RANDOM_N = 10


class TsvParamNet(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(values: np.ndarray, train_mask: np.ndarray):
    mean = values[train_mask].mean(axis=0)
    std = np.maximum(values[train_mask].std(axis=0), 1e-12)
    return (values - mean) / std, mean, std


def split_masks(df: pd.DataFrame):
    rng = np.random.default_rng(RANDOM_SEED)
    ids = df["dut_index"].to_numpy()
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * TRAIN_RATIO))
    train_ids = set(shuffled[:n_train])
    train = df["dut_index"].isin(train_ids).to_numpy()
    return train, ~train


def load_targets(df: pd.DataFrame):
    targets = []
    freq = None
    for path in df["snp_path"]:
        nw = rf.Network(str(path))
        if freq is None:
            freq = nw.f
        elif len(freq) != len(nw.f) or not np.allclose(freq, nw.f):
            raise ValueError(f"Frequency grid mismatch: {path}")
        targets.append(nw.s)
    return np.stack(targets, axis=0), freq


def prepare_arrays(df: pd.DataFrame):
    s_target, freq = load_targets(df)
    train_mask, val_mask = split_masks(df)
    x_raw = df[FEATURES].to_numpy(dtype=np.float64)
    y_raw = df[TARGET_PARAMS].to_numpy(dtype=np.float64)
    y_log = np.log(np.maximum(y_raw, 1e-300))
    x_norm, x_mean, x_std = normalize(x_raw, train_mask)
    y_norm, y_mean, y_std = normalize(y_log, train_mask)
    metadata = {
        "device_name": "TSV_Connection2",
        "feature_columns": FEATURES,
        "target_params": TARGET_PARAMS,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_log_mean": y_mean.tolist(),
        "y_log_std": y_std.tolist(),
        "freq_hz": freq.tolist(),
        "source_param_csv": str(PARAM_CSV),
    }
    return x_norm, y_norm, y_raw, train_mask, val_mask, s_target, freq, metadata


def abcd2s_torch(a, b, c, d):
    denom = a + b / Z_REF + c * Z_REF + d + 1e-30
    s = torch.zeros((*a.shape, 2, 2), dtype=COMPLEX_DTYPE, device=a.device)
    s[..., 0, 0] = (a + b / Z_REF - c * Z_REF - d) / denom
    s[..., 0, 1] = 2.0 * (a * d - b * c) / denom
    s[..., 1, 0] = 2.0 / denom
    s[..., 1, 1] = (-a + b / Z_REF - c * Z_REF + d) / denom
    return s


def circuit_params_to_s_torch(params, length_um, freqs_hz):
    r1, r2, r3 = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    l1, l2, l3 = params[:, 3:4] * 1e-9, params[:, 4:5] * 1e-9, params[:, 5:6] * 1e-9
    cox, csi, rsi = params[:, 6:7] * 1e-12, params[:, 7:8] * 1e-12, params[:, 8:9]
    omega = torch.tensor(2.0 * np.pi * freqs_hz, dtype=REAL_DTYPE, device=params.device)[None, :]
    length_m = length_um[:, None].to(params.device) * 1e-6
    j = torch.complex(torch.tensor(0.0, dtype=REAL_DTYPE, device=params.device), torch.tensor(1.0, dtype=REAL_DTYPE, device=params.device))

    r_rlgc = (r1**2 * r2 + r1 * r2**2 + omega**2 * r1 * l2**2) / ((r1 + r2) ** 2 + omega**2 * l2**2 + 1e-30) + (
        omega**2 * l3**2 * r3
    ) / (r3**2 + omega**2 * l3**2 + 1e-30)
    l_rlgc = (r1**2 * l2) / ((r1 + r2) ** 2 + omega**2 * l2**2 + 1e-30) + l3 * r3**2 / (
        r3**2 + omega**2 * l3**2 + 1e-30
    ) + l1
    g_rlgc = (omega**2 * rsi * cox**2) / (1.0 + omega**2 * rsi**2 * (cox + csi) ** 2 + 1e-30)
    c_rlgc = (cox + omega**2 * csi * rsi**2 * cox * (cox + csi)) / (1.0 + omega**2 * rsi**2 * (cox + csi) ** 2 + 1e-30)

    z0 = torch.sqrt((r_rlgc.to(COMPLEX_DTYPE) + j * omega * l_rlgc.to(COMPLEX_DTYPE)) / (g_rlgc.to(COMPLEX_DTYPE) + j * omega * c_rlgc.to(COMPLEX_DTYPE) + 1e-300))
    gamma = torch.sqrt((r_rlgc.to(COMPLEX_DTYPE) + j * omega * l_rlgc.to(COMPLEX_DTYPE)) * (g_rlgc.to(COMPLEX_DTYPE) + j * omega * c_rlgc.to(COMPLEX_DTYPE)))
    gl = gamma * length_m.to(COMPLEX_DTYPE)
    a = torch.cosh(gl)
    b = z0 * torch.sinh(gl)
    c = torch.sinh(gl) / (z0 + 1e-300)
    d = torch.cosh(gl)
    return abcd2s_torch(a, b, c, d)


def circuit_params_to_s_np(params, length_um, freqs_hz):
    with torch.no_grad():
        return circuit_params_to_s_torch(
            torch.tensor(np.asarray(params, dtype=np.float64).reshape(1, -1), dtype=REAL_DTYPE),
            torch.tensor([float(length_um)], dtype=REAL_DTYPE),
            freqs_hz,
        ).cpu().numpy()[0]


def s_loss(pred_s, target_s):
    return torch.mean((pred_s.real - target_s.real) ** 2 + (pred_s.imag - target_s.imag) ** 2)


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def wrapped_phase_delta_deg(pred, target):
    delta = np.angle(pred) - np.angle(target)
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    return np.rad2deg(delta)


def train_model(model, df, arrays, device):
    x_norm, y_norm, _, train_mask, val_mask, s_target, freq, metadata = arrays
    x_train = torch.tensor(x_norm[train_mask], dtype=REAL_DTYPE)
    y_train = torch.tensor(y_norm[train_mask], dtype=REAL_DTYPE)
    x_val = torch.tensor(x_norm[val_mask], dtype=REAL_DTYPE, device=device)
    y_val = torch.tensor(y_norm[val_mask], dtype=REAL_DTYPE, device=device)
    length_val = torch.tensor(df.loc[val_mask, "h_tsv"].to_numpy(dtype=np.float64), dtype=REAL_DTYPE, device=device)
    s_val = torch.tensor(s_target[val_mask], dtype=COMPLEX_DTYPE, device=device)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, PARAM_EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        train_loss = total / max(seen, 1)
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(x_val) - y_val) ** 2).item()
        scheduler.step(val_loss)
        history.append({"stage": "param_pretrain", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[param] epoch={epoch}, train={train_loss:.6e}, val={val_loss:.6e}", flush=True)
        if stale >= PARAM_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    y_mean = torch.tensor(metadata["y_log_mean"], dtype=REAL_DTYPE, device=device)
    y_std = torch.tensor(metadata["y_log_std"], dtype=REAL_DTYPE, device=device)
    train_ds = TensorDataset(
        torch.tensor(x_norm[train_mask], dtype=REAL_DTYPE),
        torch.tensor(y_norm[train_mask], dtype=REAL_DTYPE),
        torch.tensor(df.loc[train_mask, "h_tsv"].to_numpy(dtype=np.float64), dtype=REAL_DTYPE),
        torch.tensor(s_target[train_mask], dtype=COMPLEX_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPARAM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=35, factor=0.5)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    for epoch in range(1, SPARAM_EPOCHS + 1):
        model.train()
        total_s = 0.0
        total_anchor = 0.0
        seen = 0
        for xb, yb, lb, sb in loader:
            xb, yb, lb, sb = xb.to(device), yb.to(device), lb.to(device), sb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(xb)
            params = torch.exp(torch.clamp(pred_norm * y_std + y_mean, min=-40.0, max=40.0))
            pred_s = circuit_params_to_s_torch(params, lb, freq)
            loss_s = s_loss(pred_s, sb)
            loss_anchor = torch.mean((pred_norm - yb) ** 2)
            loss = loss_s + PARAM_ANCHOR_WEIGHT * loss_anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_s += float(loss_s.detach().cpu()) * len(xb)
            total_anchor += float(loss_anchor.detach().cpu()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_norm = model(x_val)
            val_params = torch.exp(torch.clamp(val_norm * y_std + y_mean, min=-40.0, max=40.0))
            val_pred_s = circuit_params_to_s_torch(val_params, length_val, freq)
            val_s_loss = s_loss(val_pred_s, s_val).item()
            val_anchor = torch.mean((val_norm - y_val) ** 2).item()
            val_loss = val_s_loss + PARAM_ANCHOR_WEIGHT * val_anchor
        scheduler.step(val_loss)
        history.append(
            {
                "stage": "sparam_finetune",
                "epoch": epoch,
                "train_s_loss": total_s / max(seen, 1),
                "train_anchor_loss": total_anchor / max(seen, 1),
                "val_s_loss": val_s_loss,
                "val_anchor_loss": val_anchor,
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[sparam] epoch={epoch}, train_s={total_s / max(seen, 1):.6e}, val_s={val_s_loss:.6e}", flush=True)
        if stale >= SPARAM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(history)


def evaluate(model, df, arrays, device):
    x_norm, _, y_raw, train_mask, val_mask, s_target, freq, metadata = arrays
    y_mean = torch.tensor(metadata["y_log_mean"], dtype=REAL_DTYPE, device=device)
    y_std = torch.tensor(metadata["y_log_std"], dtype=REAL_DTYPE, device=device)
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(x_norm, dtype=REAL_DTYPE, device=device))
        pred_params = torch.exp(torch.clamp(pred_norm * y_std + y_mean, min=-40.0, max=40.0)).cpu().numpy()
    metric_rows = []
    pred_rows = []
    for i, row in df.iterrows():
        pred_s = circuit_params_to_s_np(pred_params[i], row["h_tsv"], freq)
        target = s_target[i]
        split = "train" if train_mask[i] else "val"
        metric_rows.append(
            {
                "dut_index": int(row["dut_index"]),
                "split": split,
                "s_mse": float(np.mean(np.abs(pred_s - target) ** 2)),
                "s11_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 0]) - db20(target[:, 0, 0])))),
                "s21_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(target[:, 1, 0])))),
                "s11_phase_mae_deg": float(np.mean(np.abs(wrapped_phase_delta_deg(pred_s[:, 0, 0], target[:, 0, 0])))),
                "s21_phase_mae_deg": float(np.mean(np.abs(wrapped_phase_delta_deg(pred_s[:, 1, 0], target[:, 1, 0])))),
            }
        )
        pred_row = {"dut_index": int(row["dut_index"]), "split": split}
        for j, name in enumerate(TARGET_PARAMS):
            pred_row[f"target_{name}"] = float(y_raw[i, j])
            pred_row[f"pred_{name}"] = float(pred_params[i, j])
        pred_rows.append(pred_row)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def save_plots(metrics: pd.DataFrame, df: pd.DataFrame, pred_df: pd.DataFrame, s_target: np.ndarray, freq: np.ndarray):
    plot_dir = OUTPUT_DIR / "plots"
    selected = [
        ("random_val", metrics[metrics["split"].eq("val")].sample(n=min(PLOT_RANDOM_N, int(metrics["split"].eq("val").sum())), random_state=RANDOM_SEED)),
        ("worst_val", metrics[metrics["split"].eq("val")].sort_values("s_mse", ascending=False).head(PLOT_WORST_N)),
    ]
    for group, sub in selected:
        for _, metric in sub.iterrows():
            i = int(df.index[df["dut_index"].eq(metric["dut_index"])][0])
            pred_params = pred_df[pred_df["dut_index"].eq(metric["dut_index"])][[f"pred_{name}" for name in TARGET_PARAMS]].to_numpy(dtype=np.float64)[0]
            pred_s = circuit_params_to_s_np(pred_params, df.loc[i, "h_tsv"], freq)
            target = s_target[i]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
            fig.suptitle(f"TSV Connection2 dut{int(metric['dut_index'])} | {group}", x=0.02, y=0.98, ha="left")
            specs = [
                ("S11 magnitude (dB)", lambda s: db20(s[:, 0, 0])),
                ("S21 magnitude (dB)", lambda s: db20(s[:, 1, 0])),
                ("S11 phase (deg)", lambda s: np.rad2deg(np.unwrap(np.angle(s[:, 0, 0])))),
                ("S21 phase (deg)", lambda s: np.rad2deg(np.unwrap(np.angle(s[:, 1, 0])))),
            ]
            for ax, (title, fn) in zip(axes.ravel(), specs):
                ax.plot(freq / 1e9, fn(target), label="HFSS", color="black", linewidth=1.8)
                ax.plot(freq / 1e9, fn(pred_s), label="New TSV model", color="#dc2626", linestyle="--", linewidth=1.5)
                ax.set_title(title)
                ax.set_xlabel("Frequency (GHz)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            out = plot_dir / group / f"TSV_connection2_dut{int(metric['dut_index'])}_{group}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(out)
            plt.close(fig)
    return plot_dir


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("split", as_index=False)
        .agg(
            count=("dut_index", "count"),
            s_mse_mean=("s_mse", "mean"),
            s_mse_median=("s_mse", "median"),
            s11_db_mae_mean=("s11_db_mae", "mean"),
            s21_db_mae_mean=("s21_db_mae", "mean"),
            s11_phase_mae_deg_mean=("s11_phase_mae_deg", "mean"),
            s21_phase_mae_deg_mean=("s21_phase_mae_deg", "mean"),
        )
        .sort_values("split")
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PARAM_CSV.exists():
        raise FileNotFoundError(f"Missing extracted TSV parameter CSV: {PARAM_CSV}")
    df = pd.read_csv(PARAM_CSV, encoding="utf-8-sig").sort_values("dut_index").reset_index(drop=True)
    arrays = prepare_arrays(df)
    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"device={device}, samples={len(df)}", flush=True)
    model = TsvParamNet().to(dtype=REAL_DTYPE, device=device)
    history = train_model(model, df, arrays, device)
    metrics, pred_df = evaluate(model, df, arrays, device)
    summary = summarize(metrics)
    plot_dir = save_plots(metrics, df, pred_df, arrays[5], arrays[6])

    torch.save({"model_state_dict": model.state_dict(), "metadata": arrays[7]}, OUTPUT_DIR / "tsv_connection2_sparam_net.pt")
    history.to_csv(OUTPUT_DIR / "tsv_connection2_training_history.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "tsv_connection2_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUTPUT_DIR / "tsv_connection2_param_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "tsv_connection2_sparam_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "entry": Path(__file__).name,
        "output_dir": str(OUTPUT_DIR),
        "param_csv": str(PARAM_CSV),
        "samples": int(len(df)),
        "train_count": int((metrics["split"] == "train").sum()),
        "val_count": int((metrics["split"] == "val").sum()),
        "epochs_param": int(history[history["stage"].eq("param_pretrain")]["epoch"].max()),
        "epochs_sparam": int(history[history["stage"].eq("sparam_finetune")]["epoch"].max()),
        "summary": summary.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
    }
    (OUTPUT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 TSV Connection2 S-Parameter Model Training",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Parameter CSV: `{PARAM_CSV}`",
                f"- Output: `{OUTPUT_DIR}`",
                f"- Model: `{OUTPUT_DIR / 'tsv_connection2_sparam_net.pt'}`",
                f"- Plots: `{plot_dir}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
