# -*- coding: utf-8 -*-
"""Train v11 multi-head connection NNs with S-parameter smoothness loss.

Run this file directly in VS Code. No command-line arguments are required.

This entry uses the ADS-length-0.9 good-start run to define output scaling,
excludes samples whose previous shared-parameter NN prediction has sharp
resonance-like spikes, then trains a 12-position multi-head connection network
using only cascaded S11/S21 values and frequency-difference terms as the
optimization target.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


THIS_DIR = Path(__file__).resolve().parent
NN_SOURCE_SCRIPT = THIS_DIR / "train_v11_shared7_param_nns_from_all_goodstart.py"
JOINT_SOURCE_SCRIPT = THIS_DIR / "train_v11_shared7_param_sparam_joint_adslen09.py"
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
OPT_RESULT_LABEL = "v11_sharedopt_c30_adslen09_goodstart_bad"
OPT_TARGET_FILE = "v08_shared_adslen09_goodstart_bad_targets.csv"
SOURCE_ADS_LABEL = "v11_sharedopt_c30_adslen09"
PREVIOUS_JOINT_LABEL = "v11_shared7_param_sparam_joint_adslen09"
RUN_LABEL = "v11_multihead_exclude_resonance_sparam_smooth_adslen09"
ADS_DEVICE_LENGTH_SCALE = 0.9

RESONANCE_DB_D1_THRESHOLD = 12.0
RESONANCE_RI_D1_THRESHOLD = 0.2
JOINT_EPOCHS = 260
JOINT_PATIENCE = 40
JOINT_LR = 3e-5
S_D1_WEIGHT = 0.35
S_D2_WEIGHT = 0.08
BATCH_SIZE = 8
PRINT_EVERY = 10
SIGN_EPS = 1e-12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def db20(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def s11_s21_ri_np(s_params: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            s_params[:, 0, 0].real,
            s_params[:, 0, 0].imag,
            s_params[:, 1, 0].real,
            s_params[:, 1, 0].imag,
        ]
    )


def s11_s21_ri_torch(s_params):
    s11 = s_params[..., 0, 0]
    s21 = s_params[..., 1, 0]
    return torch.stack([s11.real, s11.imag, s21.real, s21.imag], dim=-1)


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    train = df["split"].eq("train").to_numpy()
    val = df["split"].eq("val").to_numpy()
    if not val.any():
        val = train.copy()
    return {"train": train, "val": val, "test": df["split"].eq("test").to_numpy()}


def normalize_by_train(values: np.ndarray, train_mask: np.ndarray):
    mean = values[train_mask].mean(axis=0)
    std = np.maximum(values[train_mask].std(axis=0), 1e-12)
    return (values - mean) / std, mean, std


def repeat_shared_targets(targets: pd.DataFrame, wrapper) -> np.ndarray:
    shared = targets[wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    return np.asarray([wrapper.repeat_shared_params(row) for row in shared], dtype=np.float64)


def multihead_target_columns(wrapper) -> list[str]:
    return [f"conn{idx}_{name}" for idx in range(1, wrapper.CONNECTION_COUNT + 1) for name in wrapper.V08_PARAM_NAMES]


def target_ri_scale(base, target_s: np.ndarray, train_mask: np.ndarray, device):
    y = np.stack(
        [
            target_s[:, :, 0, 0].real,
            target_s[:, :, 0, 0].imag,
            target_s[:, :, 1, 0].real,
            target_s[:, :, 1, 0].imag,
        ],
        axis=-1,
    )
    scale = np.maximum(y[train_mask].reshape(-1, 4).std(axis=0), 1e-12)
    return torch.tensor(scale, dtype=base.REAL_DTYPE, device=device)


def compute_resonance_table(base, wrapper, dut_df, sim, previous_pred: pd.DataFrame) -> pd.DataFrame:
    omega = 2.0 * np.pi * sim.freq_hz
    prev = previous_pred.set_index("sample_id")
    rows = []
    for i, row in dut_df.iterrows():
        sample_id = str(row["sample_id"])
        params = prev.loc[sample_id, [f"pred_{name}" for name in wrapper.V08_PARAM_NAMES]].to_numpy(dtype=np.float64)
        pred_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, params))
        ri = s11_s21_ri_np(pred_s)
        db = np.column_stack([db20(pred_s[:, 0, 0]), db20(pred_s[:, 1, 0])])
        nn_max_d1 = float(np.max(np.abs(np.diff(ri, axis=0))))
        nn_max_d2 = float(np.max(np.abs(np.diff(ri, n=2, axis=0))))
        nn_db_d1 = float(np.max(np.abs(np.diff(db, axis=0))))
        nn_db_d2 = float(np.max(np.abs(np.diff(db, n=2, axis=0))))
        is_resonant = bool(nn_db_d1 > RESONANCE_DB_D1_THRESHOLD or nn_max_d1 > RESONANCE_RI_D1_THRESHOLD)
        rows.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "nn_max_d1": nn_max_d1,
                "nn_max_d2": nn_max_d2,
                "nn_db_d1": nn_db_d1,
                "nn_db_d2": nn_db_d2,
                "is_resonant": is_resonant,
                "reason": "nn_db_d1" if nn_db_d1 > RESONANCE_DB_D1_THRESHOLD else ("nn_max_d1" if nn_max_d1 > RESONANCE_RI_D1_THRESHOLD else ""),
            }
        )
    return pd.DataFrame(rows)


def train_shared_param_model(base, wrapper, model, x_norm, y_shared_norm, masks, device):
    train_ds = TensorDataset(
        torch.tensor(x_norm[masks["train"]], dtype=base.REAL_DTYPE),
        torch.tensor(y_shared_norm[masks["train"]], dtype=base.REAL_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_x = torch.tensor(x_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    val_y = torch.tensor(y_shared_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAM_LR, weight_decay=1e-8)
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_val = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, PARAM_EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(val_x) - val_y) ** 2).item()
        train_loss = total / max(seen, 1)
        rows.append({"stage": "shared_param_pretrain", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[shared-param] epoch={epoch}, train={train_loss:.4e}, val={val_loss:.4e}", flush=True)
        if stale >= PARAM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def train_multihead_param_model(base, model, x_norm, y_multi_norm, masks, device):
    train_ds = TensorDataset(
        torch.tensor(x_norm[masks["train"]], dtype=base.REAL_DTYPE),
        torch.tensor(y_multi_norm[masks["train"]], dtype=base.REAL_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_x = torch.tensor(x_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    val_y = torch.tensor(y_multi_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAM_LR, weight_decay=1e-8)
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_val = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, PARAM_EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(val_x) - val_y) ** 2).item()
        train_loss = total / max(seen, 1)
        rows.append({"stage": "multihead_param_pretrain", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[multihead-param] epoch={epoch}, train={train_loss:.4e}, val={val_loss:.4e}", flush=True)
        if stale >= PARAM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def multihead_sparam_loss(pred_s, target_s, ri_scale):
    pred_ri = s11_s21_ri_torch(pred_s)
    target_ri = s11_s21_ri_torch(target_s)
    ri_loss = torch.mean(((pred_ri - target_ri) / ri_scale) ** 2)
    pred_d1 = pred_ri[:, 1:, :] - pred_ri[:, :-1, :]
    target_d1 = target_ri[:, 1:, :] - target_ri[:, :-1, :]
    d1_loss = torch.mean(((pred_d1 - target_d1) / ri_scale) ** 2)
    pred_d2 = pred_d1[:, 1:, :] - pred_d1[:, :-1, :]
    target_d2 = target_d1[:, 1:, :] - target_d1[:, :-1, :]
    d2_loss = torch.mean(((pred_d2 - target_d2) / ri_scale) ** 2)
    total = ri_loss + S_D1_WEIGHT * d1_loss + S_D2_WEIGHT * d2_loss
    return total, ri_loss, d1_loss, d2_loss


def train_multihead_sparam_only(base, wrapper, model, x_norm, masks, sim, y_mean, y_std, device):
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["val"])[0]
    train_ds = TensorDataset(
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(x_norm[train_idx], dtype=base.REAL_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    omega_t = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    ri_scale = target_ri_scale(base, sim.target_s, masks["train"], device)
    val_x = torch.tensor(x_norm[val_idx], dtype=base.REAL_DTYPE, device=device)
    val_base = torch.tensor(sim.base_abcds[val_idx], dtype=base.COMPLEX_DTYPE, device=device)
    val_target = torch.tensor(sim.target_s[val_idx], dtype=base.COMPLEX_DTYPE, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=JOINT_LR, weight_decay=1e-8)
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_val = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, JOINT_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for idx_b, xb in loader:
            idx_np = idx_b.numpy()
            xb = xb.to(device)
            base_b = torch.tensor(sim.base_abcds[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            target_b = torch.tensor(sim.target_s[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(xb)
            p_flat = base.denormalize_params(pred_norm, y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            pred_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t))
            loss, ri_loss, d1_loss, d2_loss = multihead_sparam_loss(pred_s, target_b, ri_scale)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            n = len(xb)
            total_loss += float(loss.detach().cpu()) * n
            seen += n
        model.eval()
        with torch.no_grad():
            val_norm = model(val_x)
            val_flat = base.denormalize_params(val_norm, y_mean_t, y_std_t)
            val_all = val_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            val_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, val_base, val_all, omega_t))
            val_loss, val_ri, val_d1, val_d2 = multihead_sparam_loss(val_s, val_target, ri_scale)
        row = {
            "stage": "multihead_sparam_smooth",
            "epoch": epoch,
            "train_total_loss": float(total_loss / max(seen, 1)),
            "train_ri_loss": float(total_loss / max(seen, 1)),
            "val_total_loss": float(val_loss.detach().cpu()),
            "val_ri_loss": float(val_ri.detach().cpu()),
            "val_d1_loss": float(val_d1.detach().cpu()),
            "val_d2_loss": float(val_d2.detach().cpu()),
        }
        rows.append(row)
        if row["val_total_loss"] < best_val:
            best_val = row["val_total_loss"]
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(
                f"[multihead-sparam-smooth] epoch={epoch}, train={row['train_total_loss']:.4e}, "
                f"val={row['val_total_loss']:.4e}, val_ri={row['val_ri_loss']:.4e}, "
                f"val_d1={row['val_d1_loss']:.4e}",
                flush=True,
            )
        if stale >= JOINT_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def metric_dict(base, target_s: np.ndarray, pred_s: np.ndarray) -> dict[str, float]:
    return {
        "mse_all_s": float(np.mean(np.abs(pred_s - target_s) ** 2)),
        "nmse_s11_s21_ri": base.nmse_s11_s21_real_imag(target_s, pred_s),
        "mag_phase_mse_s11_s21": base.s11_s21_mag_phase_mse_np(target_s, pred_s),
        "s11_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 0]) - db20(target_s[:, 0, 0])))),
        "s21_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(target_s[:, 1, 0])))),
    }


def evaluate_multihead(base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_mean, y_std, device):
    omega = 2.0 * np.pi * sim.freq_hz
    omega_t = torch.tensor(omega, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    opt_by_id = opt_targets.set_index("sample_id")
    rows = []
    pred_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dut_df), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(dut_df))
            x_b = torch.tensor(x_norm[start:stop], dtype=base.REAL_DTYPE, device=device)
            base_b = torch.tensor(sim.base_abcds[start:stop], dtype=base.COMPLEX_DTYPE, device=device)
            p_flat = base.denormalize_params(model(x_b), y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            pred_s_batch = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t)).cpu().numpy()
            pred_params = p_flat.cpu().numpy()
            for local_i in range(stop - start):
                i = start + local_i
                sample = dut_df.iloc[i]
                sample_id = str(sample["sample_id"])
                target_s = sim.target_s[i]
                direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
                opt_p = opt_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
                opt_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, opt_p))
                nn_s = pred_s_batch[local_i]
                direct = metric_dict(base, target_s, direct_s)
                opt = metric_dict(base, target_s, opt_s)
                nn = metric_dict(base, target_s, nn_s)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": sample["split"],
                        "file": sample["file"],
                        "dut_index": int(sample["dut_index"]),
                        "direct_nmse_s11_s21_ri": direct["nmse_s11_s21_ri"],
                        "optimized_nmse_s11_s21_ri": opt["nmse_s11_s21_ri"],
                        "nn_nmse_s11_s21_ri": nn["nmse_s11_s21_ri"],
                        "direct_mse_all_s": direct["mse_all_s"],
                        "optimized_mse_all_s": opt["mse_all_s"],
                        "nn_mse_all_s": nn["mse_all_s"],
                        "direct_s11_db_mae": direct["s11_db_mae"],
                        "optimized_s11_db_mae": opt["s11_db_mae"],
                        "nn_s11_db_mae": nn["s11_db_mae"],
                        "direct_s21_db_mae": direct["s21_db_mae"],
                        "optimized_s21_db_mae": opt["s21_db_mae"],
                        "nn_s21_db_mae": nn["s21_db_mae"],
                    }
                )
                pred_row = {"sample_id": sample_id, "split": sample["split"]}
                for col_idx, col_name in enumerate(multihead_target_columns(wrapper)):
                    pred_row[f"pred_{col_name}"] = float(pred_params[local_i, col_idx])
                pred_rows.append(pred_row)
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def summary_row(split: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "split": split,
        "count": int(len(group)),
        "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
        "optimized_nmse_mean": float(group["optimized_nmse_s11_s21_ri"].mean()),
        "nn_nmse_mean": float(group["nn_nmse_s11_s21_ri"].mean()),
        "optimized_nmse_median": float(group["optimized_nmse_s11_s21_ri"].median()),
        "nn_nmse_median": float(group["nn_nmse_s11_s21_ri"].median()),
        "nn_better_than_direct_count": int((group["nn_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()),
        "nn_better_than_optimized_count": int((group["nn_nmse_s11_s21_ri"] < group["optimized_nmse_s11_s21_ri"]).sum()),
        "optimized_s11_db_mae_mean": float(group["optimized_s11_db_mae"].mean()),
        "nn_s11_db_mae_mean": float(group["nn_s11_db_mae"].mean()),
        "optimized_s21_db_mae_mean": float(group["optimized_s21_db_mae"].mean()),
        "nn_s21_db_mae_mean": float(group["nn_s21_db_mae"].mean()),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


def save_training_plots(base, output_dir: Path, histories: list[pd.DataFrame], summary: pd.DataFrame):
    fig, axes = base.plt.subplots(1, 2, figsize=(13, 4), dpi=150)
    for hist in histories:
        if "train_loss" in hist.columns:
            axes[0].plot(hist["epoch"], hist["train_loss"], label=f"{hist['stage'].iloc[0]} train")
            axes[0].plot(hist["epoch"], hist["val_loss"], label=f"{hist['stage'].iloc[0]} val")
        elif "train_total_loss" in hist.columns:
            axes[1].plot(hist["epoch"], hist["train_total_loss"], label="joint train")
            axes[1].plot(hist["epoch"], hist["val_total_loss"], label="joint val")
            axes[1].plot(hist["epoch"], hist["val_ri_loss"], label="joint val RI")
    axes[0].set_title("No parameter pretraining")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.5, 0.5, "S-parameter-only training", ha="center", va="center", transform=axes[0].transAxes)
    axes[1].set_title("S-parameter joint training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "multihead_training_loss.png")
    base.plt.close(fig)

    fig, ax = base.plt.subplots(figsize=(7, 4), dpi=150)
    plot_df = summary[summary["split"].ne("all")]
    x = np.arange(len(plot_df))
    width = 0.28
    ax.bar(x - width, plot_df["direct_nmse_mean"], width, label="direct")
    ax.bar(x, plot_df["optimized_nmse_mean"], width, label="optimized")
    ax.bar(x + width, plot_df["nn_nmse_mean"], width, label="multihead NN")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["split"])
    ax.set_ylabel("NMSE mean")
    ax.set_title("Filtered sample NMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "multihead_filtered_nmse_summary.png")
    base.plt.close(fig)


def save_comparison_plots(base, wrapper, output_dir: Path, dut_df, sim, metrics, opt_targets, pred_table):
    plot_dir = output_dir / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    opt_by_id = opt_targets.set_index("sample_id")
    pred_by_id = pred_table.set_index("sample_id")
    selected = pd.concat(
        [
            metrics[metrics["split"].eq("test")].sort_values("nn_nmse_s11_s21_ri", ascending=False).head(10),
            metrics[metrics["split"].eq("test")].sample(n=min(8, int(metrics["split"].eq("test").sum())), random_state=20260710),
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")
    paths = []
    for _, metric in selected.iterrows():
        sample_id = str(metric["sample_id"])
        idx = int(dut_df.index[dut_df["sample_id"].eq(sample_id)][0])
        target_s = sim.target_s[idx]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[idx])))
        opt_p = opt_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        opt_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[idx], omega, opt_p))
        pred_flat = pred_by_id.loc[sample_id, [f"pred_{col}" for col in multihead_target_columns(wrapper)]].to_numpy(dtype=np.float64)
        pred_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[idx], omega, pred_flat))
        fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
        fig.suptitle(
            f"{sample_id} | opt={metric['optimized_nmse_s11_s21_ri']:.3e} | multihead={metric['nn_nmse_s11_s21_ri']:.3e}",
            x=0.02,
            y=0.985,
            ha="left",
        )
        specs = [(0, 0, "S11 real", np.real), (0, 0, "S11 imag", np.imag), (1, 0, "S21 real", np.real), (1, 0, "S21 imag", np.imag)]
        for ax, (m, n, title, fn) in zip(axes.ravel(), specs):
            ax.plot(freq_ghz, fn(target_s[:, m, n]), label="HFSS", color="black", linewidth=1.8)
            ax.plot(freq_ghz, fn(direct_s[:, m, n]), label="ADS direct", color="#64748b", linestyle=":")
            ax.plot(freq_ghz, fn(opt_s[:, m, n]), label="Optimized shared", color="#16a34a", linestyle="--")
            ax.plot(freq_ghz, fn(pred_s[:, m, n]), label="Multi-head NN", color="#dc2626", linestyle="-.")
            ax.set_title(title)
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = plot_dir / f"{sample_id}.png"
        fig.savefig(out_path)
        base.plt.close(fig)
        paths.append(str(out_path))
    return plot_dir, paths


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_multihead_filtered")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_wrapper_for_multihead_filtered")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_multihead_filtered")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    opt_dir = version_root / "results" / OPT_RESULT_LABEL
    previous_joint_dir = version_root / "results" / PREVIOUS_JOINT_LABEL
    source_ads_dir = version_root / "results" / SOURCE_ADS_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = source_ads_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = ADS_DEVICE_LENGTH_SCALE
    base.set_seed(base.RANDOM_SEED)

    opt_targets_all = pd.read_csv(opt_dir / OPT_TARGET_FILE, encoding="utf-8-sig")
    dut_all = wrapper.collect_v11_samples(base)
    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    sim_all = base.load_single_device_simulation(dut_all, settings)
    previous_pred = pd.read_csv(previous_joint_dir / "joint_predictions.csv", encoding="utf-8-sig")
    resonance = compute_resonance_table(base, wrapper, dut_all, sim_all, previous_pred)
    resonance.to_csv(output_dir / "resonance_diagnostics_all_samples.csv", index=False, encoding="utf-8-sig")
    excluded = resonance[resonance["is_resonant"]].copy()
    excluded.to_csv(output_dir / "excluded_resonance_samples.csv", index=False, encoding="utf-8-sig")

    active_ids = set(resonance.loc[~resonance["is_resonant"], "sample_id"])
    dut_df = dut_all[dut_all["sample_id"].isin(active_ids)].reset_index(drop=True)
    opt_targets = dut_df[["sample_id"]].merge(opt_targets_all, on="sample_id", how="left")
    source_index = [int(i) for i in dut_all.index[dut_all["sample_id"].isin(active_ids)]]
    sim = copy.copy(sim_all)
    sim.base_abcds = sim_all.base_abcds[source_index]
    sim.target_s = sim_all.target_s[source_index]

    masks = split_masks(opt_targets)
    x_raw = opt_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_multi_raw = repeat_shared_targets(opt_targets, wrapper)
    x_norm, x_mean, x_std = normalize_by_train(x_raw, masks["train"])
    y_multi_norm, y_multi_mean, y_multi_std = normalize_by_train(y_multi_raw, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    multi_model = wrapper.MultiHeadV08ConnectionNet(input_dim=x_norm.shape[1]).to(dtype=base.REAL_DTYPE, device=device)
    joint_history = train_multihead_sparam_only(base, wrapper, multi_model, x_norm, masks, sim, y_multi_mean, y_multi_std, device)
    joint_metrics, joint_pred = evaluate_multihead(base, wrapper, multi_model, dut_df, sim, opt_targets, x_norm, y_multi_mean, y_multi_std, device)
    joint_summary = summarize(joint_metrics)

    save_training_plots(base, output_dir, [joint_history], joint_summary)
    plot_dir, plot_paths = save_comparison_plots(base, wrapper, output_dir, dut_df, sim, joint_metrics, opt_targets, joint_pred)

    torch.save(
        {
            "model_state_dict": multi_model.state_dict(),
            "metadata": {
            "model_type": "v11_multihead_exclude_resonance_sparam_smooth_adslen09",
                "architecture": "seven parameter trunks with 12 connection-position heads",
                "feature_columns": base.STRUCTURE_COLUMNS,
                "target_columns": multihead_target_columns(wrapper),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_multi_mean.tolist(),
                "y_std": y_multi_std.tolist(),
                "source_optimized_targets": str(opt_dir / OPT_TARGET_FILE),
                "source_ads_cache": str(base.ADS_CACHE_DIR),
                "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
                "excluded_resonance_count": int(len(excluded)),
                "active_sample_count": int(len(dut_df)),
                "resonance_filter": {
                    "nn_db_d1_threshold": RESONANCE_DB_D1_THRESHOLD,
                    "nn_ri_d1_threshold": RESONANCE_RI_D1_THRESHOLD,
                    "previous_joint_result": PREVIOUS_JOINT_LABEL,
                },
                "connection_count": wrapper.CONNECTION_COUNT,
                "v08_scale_factors": wrapper.V08_SCALE_FACTORS.tolist(),
                "freq_hz": sim.freq_hz.tolist(),
            },
        },
        output_dir / "multihead_exclude_resonance.pt",
    )

    joint_history.to_csv(output_dir / "multihead_sparam_smooth_history.csv", index=False, encoding="utf-8-sig")
    joint_metrics.to_csv(output_dir / "multihead_joint_metrics.csv", index=False, encoding="utf-8-sig")
    joint_summary.to_csv(output_dir / "multihead_joint_summary.csv", index=False, encoding="utf-8-sig")
    joint_pred.to_csv(output_dir / "multihead_joint_predictions.csv", index=False, encoding="utf-8-sig")

    filter_summary = (
        resonance.groupby(["split", "is_resonant"], as_index=False)
        .agg(count=("sample_id", "count"), nn_db_d1_mean=("nn_db_d1", "mean"), nn_max_d1_mean=("nn_max_d1", "mean"))
    )
    filter_summary.to_csv(output_dir / "resonance_filter_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "source_optimized_targets": str(opt_dir / OPT_TARGET_FILE),
        "previous_joint_result": str(previous_joint_dir),
        "excluded_resonance_count": int(len(excluded)),
        "active_sample_count": int(len(dut_df)),
        "train_val_test_active": {
            "train": int(masks["train"].sum()),
            "val": int(masks["val"].sum()),
            "test": int(masks["test"].sum()),
        },
        "resonance_filter": {
            "nn_db_d1_threshold": RESONANCE_DB_D1_THRESHOLD,
            "nn_ri_d1_threshold": RESONANCE_RI_D1_THRESHOLD,
        },
        "training_objective": "S11/S21 real/imag loss plus S-parameter frequency-difference smoothness loss",
        "s_d1_weight": S_D1_WEIGHT,
        "s_d2_weight": S_D2_WEIGHT,
        "joint_summary": joint_summary.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_paths,
    }
    (output_dir / "multihead_exclude_resonance_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "multihead_exclude_resonance_report.md").write_text(
        "\n".join(
            [
                "# V11 Multi-Head S-Parameter Smoothness Training With Resonance Exclusion",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source optimized targets: `{opt_dir / OPT_TARGET_FILE}`",
                f"- Resonance source: previous joint NN predictions in `{previous_joint_dir}`",
                f"- Resonance filter: `nn_db_d1 > {RESONANCE_DB_D1_THRESHOLD}` or `nn_max_d1 > {RESONANCE_RI_D1_THRESHOLD}`",
                f"- Training objective: cascaded `S11/S21` real/imag loss + `{S_D1_WEIGHT}` first-difference loss + `{S_D2_WEIGHT}` second-difference loss",
                f"- Excluded samples: `{len(excluded)}`",
                f"- Active samples: `{len(dut_df)}`",
                f"- Active train/val/test: `{int(masks['train'].sum())}` / `{int(masks['val'].sum())}` / `{int(masks['test'].sum())}`",
                "",
                "## Resonance Filter Summary",
                "",
                dataframe_to_markdown(filter_summary),
                "",
                "## Multi-Head S-Parameter Smoothness Summary",
                "",
                dataframe_to_markdown(joint_summary),
                "",
                "## Outputs",
                "",
                f"- Checkpoint: `{output_dir / 'multihead_exclude_resonance.pt'}`",
                f"- Excluded samples: `{output_dir / 'excluded_resonance_samples.csv'}`",
                f"- Joint metrics: `{output_dir / 'multihead_joint_metrics.csv'}`",
                f"- Joint summary: `{output_dir / 'multihead_joint_summary.csv'}`",
                f"- Joint predictions: `{output_dir / 'multihead_joint_predictions.csv'}`",
                f"- Training plot: `{output_dir / 'multihead_training_loss.png'}`",
                f"- NMSE plot: `{output_dir / 'multihead_filtered_nmse_summary.png'}`",
                f"- Comparison plots: `{plot_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "validation_archive.md").write_text(
        "\n".join(
            [
                "# Validation Archive",
                "",
                f"- Entry: `{Path(__file__).name}`",
                "- Status: completed",
                f"- Source samples: `{len(dut_all)}`",
                f"- Excluded resonant samples: `{len(excluded)}`",
                f"- Active samples: `{len(dut_df)}`",
                f"- Active train/val/test: `{int(masks['train'].sum())}` / `{int(masks['val'].sum())}` / `{int(masks['test'].sum())}`",
                f"- Source ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Comparison plots: `{len(plot_paths)}`",
                "",
                "## Multi-Head Joint Summary",
                "",
                dataframe_to_markdown(joint_summary),
            ]
        ),
        encoding="utf-8",
    )
    print("Resonance filter summary:", flush=True)
    print(dataframe_to_markdown(filter_summary), flush=True)
    print("Multi-head joint summary:", flush=True)
    print(dataframe_to_markdown(joint_summary), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
