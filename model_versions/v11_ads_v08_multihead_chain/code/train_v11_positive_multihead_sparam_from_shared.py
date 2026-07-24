# -*- coding: utf-8 -*-
"""Expand the positive shared parameter NN to multi-head and train on S-params.

Run this file directly in VS Code. No command-line arguments are required.

The initial weights and biases are copied from the current seven-network
``input -> 30 -> 30 -> 20 -> 1`` model. The multi-head model has 12 connection
heads per circuit parameter. Training uses only cascaded S11/S21 real/imag loss.
Outputs are log10(parameters), converted back to positive circuit parameters
inside the cascade.
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
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
POSITIVE_SCRIPT = THIS_DIR / "optimize_v11_positive_shared_connection_lhs400_adslen09.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"

OPT_RESULT_LABEL = "v11_positive_sharedopt_lhs400_connection2_goodstart_all_adslen09"
OPT_TARGET_FILE = "v08_positive_goodstart_targets.csv"
SHARED_NN_LABEL = "v11_positive_goodstart_shared7_param_nns_log_adslen09"
SOURCE_ADS_LABEL = "v11_positive_sharedopt_lhs400_connection2_adslen09"
RUN_LABEL = "v11_positive_multihead_sparam_from_shared_log_adslen09"

ADS_DEVICE_LENGTH_SCALE = 0.9
POSITIVE_LOWER = 1e-9
POSITIVE_UPPER = 1e5
LOG_LOWER = -9.0
LOG_UPPER = 5.0
JOINT_EPOCHS = 320
JOINT_PATIENCE = 55
JOINT_LR = 2e-5
BATCH_SIZE = 8
PRINT_EVERY = 10
PLOT_WORST_VAL = 12
PLOT_WORST_ALL = 12
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


def repeat_shared_targets(values: np.ndarray, wrapper) -> np.ndarray:
    return np.asarray([wrapper.repeat_shared_params(row) for row in values], dtype=np.float64)


def multihead_target_columns(wrapper) -> list[str]:
    return [f"conn{idx}_{name}" for idx in range(1, wrapper.CONNECTION_COUNT + 1) for name in wrapper.V08_PARAM_NAMES]


def s11_s21_ri_torch(s_params):
    s11 = s_params[..., 0, 0]
    s21 = s_params[..., 1, 0]
    return torch.stack([s11.real, s11.imag, s21.real, s21.imag], dim=-1)


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


def denorm_log_to_positive_params(base, pred_norm, y_mean_t, y_std_t, connection_count: int, param_count: int):
    log_flat = base.denormalize_params(pred_norm, y_mean_t, y_std_t)
    log_flat = torch.clamp(log_flat, LOG_LOWER, LOG_UPPER)
    p_flat = torch.pow(torch.tensor(10.0, dtype=base.REAL_DTYPE, device=pred_norm.device), log_flat)
    p_flat = torch.clamp(p_flat, POSITIVE_LOWER, POSITIVE_UPPER)
    return p_flat, p_flat.reshape(-1, connection_count, param_count), log_flat


def multihead_sparam_loss(pred_s, target_s, ri_scale):
    pred_ri = s11_s21_ri_torch(pred_s)
    target_ri = s11_s21_ri_torch(target_s)
    return torch.mean(((pred_ri - target_ri) / ri_scale) ** 2)


def db20(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def metric_dict(base, target_s: np.ndarray, pred_s: np.ndarray) -> dict[str, float]:
    return {
        "mse_all_s": float(np.mean(np.abs(pred_s - target_s) ** 2)),
        "nmse_s11_s21_ri": base.nmse_s11_s21_real_imag(target_s, pred_s),
        "mag_phase_mse_s11_s21": base.s11_s21_mag_phase_mse_np(target_s, pred_s),
        "s11_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 0]) - db20(target_s[:, 0, 0])))),
        "s21_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(target_s[:, 1, 0])))),
    }


def parameter_sign_stats(pred_table: pd.DataFrame, wrapper) -> pd.DataFrame:
    rows = []
    for name in wrapper.V08_PARAM_NAMES:
        cols = [f"pred_conn{idx}_{name}" for idx in range(1, wrapper.CONNECTION_COUNT + 1)]
        values = pred_table[cols].to_numpy(dtype=np.float64).ravel()
        rows.append(
            {
                "parameter": name,
                "count": int(len(values)),
                "negative_count": int(np.sum(values < -SIGN_EPS)),
                "zero_count": int(np.sum(np.abs(values) <= SIGN_EPS)),
                "positive_count": int(np.sum(values > SIGN_EPS)),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def load_initialized_models(base, wrapper, shared_ckpt_path: Path, device):
    shared_model = wrapper.SharedV08ParamNet(input_dim=len(base.STRUCTURE_COLUMNS)).to(dtype=base.REAL_DTYPE, device=device)
    checkpoint = torch.load(shared_ckpt_path, map_location=device)
    shared_model.load_state_dict(checkpoint["model_state_dict"])
    multi_model = wrapper.MultiHeadV08ConnectionNet(input_dim=len(base.STRUCTURE_COLUMNS)).to(dtype=base.REAL_DTYPE, device=device)
    multi_model.initialize_from_shared(shared_model)
    return shared_model, multi_model, checkpoint


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
            p_flat, p_all, log_flat = denorm_log_to_positive_params(
                base,
                model(x_b),
                y_mean_t,
                y_std_t,
                wrapper.CONNECTION_COUNT,
                len(wrapper.V08_PARAM_NAMES),
            )
            pred_s_batch = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t)).cpu().numpy()
            pred_params = p_flat.cpu().numpy()
            pred_logs = log_flat.cpu().numpy()
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
                    pred_row[f"pred_log10_{col_name}"] = float(pred_logs[local_i, col_idx])
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


def train_sparam_only(base, wrapper, model, x_norm, masks, sim, y_mean, y_std, device):
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["val"])[0]
    train_ds = TensorDataset(torch.tensor(train_idx, dtype=torch.long), torch.tensor(x_norm[train_idx], dtype=base.REAL_DTYPE))
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
            _, p_all, _ = denorm_log_to_positive_params(base, model(xb), y_mean_t, y_std_t, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            pred_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t))
            loss = multihead_sparam_loss(pred_s, target_b, ri_scale)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(xb)
            total_loss += float(loss.detach().cpu()) * n
            seen += n
        model.eval()
        with torch.no_grad():
            _, val_all, _ = denorm_log_to_positive_params(base, model(val_x), y_mean_t, y_std_t, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            val_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, val_base, val_all, omega_t))
            val_loss = multihead_sparam_loss(val_s, val_target, ri_scale)
        row = {
            "stage": "multihead_sparam_only_from_shared",
            "epoch": epoch,
            "train_ri_loss": float(total_loss / max(seen, 1)),
            "val_ri_loss": float(val_loss.detach().cpu()),
        }
        rows.append(row)
        if row["val_ri_loss"] < best_val:
            best_val = row["val_ri_loss"]
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[multihead-sparam] epoch={epoch}, train_ri={row['train_ri_loss']:.4e}, val_ri={row['val_ri_loss']:.4e}", flush=True)
        if stale >= JOINT_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def save_plots(base, wrapper, output_dir: Path, history: pd.DataFrame, metrics: pd.DataFrame, dut_df, sim, opt_targets, pred_table):
    fig, axes = base.plt.subplots(1, 2, figsize=(13, 4), dpi=150)
    axes[0].plot(history["epoch"], history["train_ri_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_ri_loss"], label="val")
    axes[0].set_title("S-parameter-only loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalized S11/S21 RI loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].scatter(metrics["optimized_nmse_s11_s21_ri"], metrics["nn_nmse_s11_s21_ri"], s=18, alpha=0.75)
    max_nmse = float(max(metrics["optimized_nmse_s11_s21_ri"].max(), metrics["nn_nmse_s11_s21_ri"].max()))
    axes[1].plot([0, max_nmse], [0, max_nmse], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Optimized NMSE")
    axes[1].set_ylabel("Multi-head NN NMSE")
    axes[1].set_title("Optimized vs multi-head")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "multihead_sparam_training_summary.png")
    base.plt.close(fig)

    plot_dir = output_dir / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    opt_by_id = opt_targets.set_index("sample_id")
    pred_by_id = pred_table.set_index("sample_id")
    selected = pd.concat(
        [
            metrics[metrics["split"].eq("val")].sort_values("nn_nmse_s11_s21_ri", ascending=False).head(PLOT_WORST_VAL),
            metrics.sort_values("nn_nmse_s11_s21_ri", ascending=False).head(PLOT_WORST_ALL),
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
        fig.suptitle(f"{sample_id} | opt={metric['optimized_nmse_s11_s21_ri']:.3e} | multihead={metric['nn_nmse_s11_s21_ri']:.3e}", x=0.02, y=0.985, ha="left")
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
    nnsrc = load_module(NN_SOURCE_SCRIPT, "v11_positive_multihead_nn_source")
    source = load_module(SOURCE_SCRIPT, "v11_positive_multihead_calibrated_source")
    positive = load_module(POSITIVE_SCRIPT, "v11_positive_multihead_positive_source")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_positive_multihead_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_positive_multihead_base")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    opt_dir = version_root / "results" / OPT_RESULT_LABEL
    shared_nn_dir = version_root / "results" / SHARED_NN_LABEL
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

    opt_target_path = opt_dir / OPT_TARGET_FILE
    opt_targets_all = pd.read_csv(opt_target_path, encoding="utf-8-sig")
    dut_all = positive.collect_lhs400_rdl_tsv_samples(base)
    target_ids = set(opt_targets_all["sample_id"].astype(str))
    excluded_unoptimized = dut_all[~dut_all["sample_id"].astype(str).isin(target_ids)].copy()
    dut_df = dut_all[dut_all["sample_id"].astype(str).isin(target_ids)].reset_index(drop=True)
    opt_targets = dut_df[["sample_id"]].merge(opt_targets_all, on="sample_id", how="left")
    if opt_targets[wrapper.V08_PARAM_NAMES].isna().any().any():
        raise ValueError("Positive good-start target table is missing parameter values after sample alignment.")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    settings["ads_geometry_scale_note"] = "Applied to l_tmrdl, l_bsmrdl, and h_tsv by the v11 base ADS runner."
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)

    masks = split_masks(opt_targets)
    x_raw = opt_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_linear = opt_targets[wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    y_log_shared = np.log10(np.clip(y_linear, POSITIVE_LOWER, POSITIVE_UPPER))
    y_log_multi = repeat_shared_targets(y_log_shared, wrapper)
    x_norm, x_mean, x_std = normalize_by_train(x_raw, masks["train"])
    y_log_multi_norm, y_log_multi_mean, y_log_multi_std = normalize_by_train(y_log_multi, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    shared_ckpt_path = shared_nn_dir / "positive_shared7_param_nns_log.pt"
    _, model, shared_checkpoint = load_initialized_models(base, wrapper, shared_ckpt_path, device)
    initial_metrics, initial_pred = evaluate_multihead(base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_log_multi_mean, y_log_multi_std, device)
    initial_summary = summarize(initial_metrics)

    history = train_sparam_only(base, wrapper, model, x_norm, masks, sim, y_log_multi_mean, y_log_multi_std, device)
    metrics, pred_table = evaluate_multihead(base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_log_multi_mean, y_log_multi_std, device)
    summary = summarize(metrics)
    sign_stats = parameter_sign_stats(pred_table, wrapper)
    plot_dir, plot_paths = save_plots(base, wrapper, output_dir, history, metrics, dut_df, sim, opt_targets, pred_table)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": "v11_positive_multihead_sparam_from_shared_log_adslen09",
                "architecture": "seven parameter trunks with 12 connection-position heads initialized from shared 30-30-20 networks",
                "target_transform": "network outputs normalized log10 positive parameters; inverse power(10) clipped to [1e-9, 1e5]",
                "training_objective": "cascaded S11/S21 real/imag loss only",
                "feature_columns": base.STRUCTURE_COLUMNS,
                "target_columns": multihead_target_columns(wrapper),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_log_multi_mean": y_log_multi_mean.tolist(),
                "y_log_multi_std": y_log_multi_std.tolist(),
                "positive_bounds": [POSITIVE_LOWER, POSITIVE_UPPER],
                "source_shared_checkpoint": str(shared_ckpt_path),
                "source_optimized_targets": str(opt_target_path),
                "source_ads_cache": str(base.ADS_CACHE_DIR),
                "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
                "excluded_unoptimized_sample_ids": excluded_unoptimized["sample_id"].astype(str).tolist(),
                "connection_count": wrapper.CONNECTION_COUNT,
                "v08_scale_factors": wrapper.V08_SCALE_FACTORS.tolist(),
                "freq_hz": sim.freq_hz.tolist(),
            },
        },
        output_dir / "positive_multihead_sparam_from_shared.pt",
    )

    history.to_csv(output_dir / "positive_multihead_sparam_history.csv", index=False, encoding="utf-8-sig")
    initial_metrics.to_csv(output_dir / "initial_shared_expanded_metrics.csv", index=False, encoding="utf-8-sig")
    initial_summary.to_csv(output_dir / "initial_shared_expanded_summary.csv", index=False, encoding="utf-8-sig")
    initial_pred.to_csv(output_dir / "initial_shared_expanded_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "positive_multihead_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "positive_multihead_sparam_summary.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(output_dir / "positive_multihead_sparam_predictions.csv", index=False, encoding="utf-8-sig")
    sign_stats.to_csv(output_dir / "positive_multihead_parameter_sign_stats.csv", index=False, encoding="utf-8-sig")
    excluded_unoptimized.to_csv(output_dir / "excluded_unoptimized_samples.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "source_shared_checkpoint": str(shared_ckpt_path),
        "source_optimized_targets": str(opt_target_path),
        "source_ads_cache": str(base.ADS_CACHE_DIR),
        "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
        "training_objective": "S11/S21 real/imag loss only",
        "samples": int(len(dut_df)),
        "excluded_unoptimized_samples": int(len(excluded_unoptimized)),
        "epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
        "best_val_ri_loss": float(history["val_ri_loss"].min()) if len(history) else None,
        "initial_summary": initial_summary.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "parameter_sign_stats": sign_stats.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_paths,
        "shared_checkpoint_metadata": shared_checkpoint.get("metadata", {}),
    }
    (output_dir / "positive_multihead_sparam_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "positive_multihead_sparam_report.md").write_text(
        "\n".join(
            [
                "# V11 Positive Multi-Head S-Parameter Training From Shared NN",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source shared checkpoint: `{shared_ckpt_path}`",
                f"- Source optimized targets: `{opt_target_path}`",
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}`",
                f"- Samples used: `{len(dut_df)}`; unoptimized current-disk samples excluded: `{len(excluded_unoptimized)}`.",
                "- Initialization: copy the current shared `input -> 30 -> 30 -> 20 -> 1` network weights and biases into all 12 multi-head positions.",
                "- Training target: cascaded `S11/S21` real/imag loss only.",
                "- Output constraint: model outputs `log10(parameter)`, converted back to positive scale and clipped to `[1e-9, 1e5]`.",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Best validation RI loss: `{report['best_val_ri_loss']}`",
                "",
                "## Initial Shared-Expanded Summary",
                "",
                dataframe_to_markdown(initial_summary),
                "",
                "## S-Parameter-Trained Multi-Head Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
                "",
                "## Outputs",
                "",
                f"- Checkpoint: `{output_dir / 'positive_multihead_sparam_from_shared.pt'}`",
                f"- Training history: `{output_dir / 'positive_multihead_sparam_history.csv'}`",
                f"- Metrics: `{output_dir / 'positive_multihead_sparam_metrics.csv'}`",
                f"- Summary: `{output_dir / 'positive_multihead_sparam_summary.csv'}`",
                f"- Predictions: `{output_dir / 'positive_multihead_sparam_predictions.csv'}`",
                f"- Sign stats: `{output_dir / 'positive_multihead_parameter_sign_stats.csv'}`",
                f"- Training plot: `{output_dir / 'multihead_sparam_training_summary.png'}`",
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
                f"- Samples: `{len(dut_df)}`",
                f"- Train/val/test: `{int(masks['train'].sum())}` / `{int(masks['val'].sum())}` / `{int(masks['test'].sum())}`",
                f"- Unoptimized current-disk samples excluded: `{len(excluded_unoptimized)}`",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Source ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Comparison plots: `{len(plot_paths)}`",
                f"- Predicted parameter nonpositive total: `{int(sign_stats['negative_count'].sum() + sign_stats['zero_count'].sum())}`",
                "",
                "## Initial Shared-Expanded Summary",
                "",
                dataframe_to_markdown(initial_summary),
                "",
                "## S-Parameter-Trained Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
            ]
        ),
        encoding="utf-8",
    )
    print("Initial shared-expanded summary:", flush=True)
    print(dataframe_to_markdown(initial_summary), flush=True)
    print("S-parameter-trained multi-head summary:", flush=True)
    print(dataframe_to_markdown(summary), flush=True)
    print("Parameter sign summary:", flush=True)
    print(dataframe_to_markdown(sign_stats), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
