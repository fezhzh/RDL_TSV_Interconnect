# -*- coding: utf-8 -*-
"""Train v11 shared 7-parameter NNs with parameter and S-parameter losses.

Run this file directly in VS Code. No command-line arguments are required.

This entry uses the ADS-length-0.9 good-start optimized circuit parameters as
parameter supervision, then continues training the same shared 7-parameter
network with an additional differentiable cascade S-parameter loss.
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
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
OPT_RESULT_LABEL = "v11_sharedopt_c30_adslen09_goodstart_bad"
OPT_TARGET_FILE = "v08_shared_adslen09_goodstart_bad_targets.csv"
SOURCE_ADS_LABEL = "v11_sharedopt_c30_adslen09"
RUN_LABEL = "v11_shared7_param_sparam_joint_adslen09"
ADS_DEVICE_LENGTH_SCALE = 0.9

JOINT_EPOCHS = 220
JOINT_PATIENCE = 45
JOINT_LR = 3e-5
PARAM_ANCHOR_WEIGHT = 0.15
MAG_PHASE_WEIGHT = 0.02
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


def cascade_shared_model(base, wrapper, model, xb, base_b, omega_t, y_mean_t, y_std_t):
    pred_norm = model(xb)
    p_shared = base.denormalize_params(pred_norm, y_mean_t, y_std_t)
    p_all = p_shared[:, None, :].repeat(1, wrapper.CONNECTION_COUNT, 1)
    pred_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t))
    return pred_norm, p_shared, pred_s


def joint_loss(base, pred_norm, target_norm, pred_s, target_s, ri_scale):
    param_loss = torch.mean((pred_norm - target_norm) ** 2)
    pred_ri = s11_s21_ri_torch(pred_s)
    target_ri = s11_s21_ri_torch(target_s)
    ri_loss = torch.mean(((pred_ri - target_ri) / ri_scale) ** 2)
    mag_phase_loss = base.s11_s21_mag_phase_loss_torch(pred_s, target_s)
    total = ri_loss + PARAM_ANCHOR_WEIGHT * param_loss + MAG_PHASE_WEIGHT * mag_phase_loss
    return total, param_loss, ri_loss, mag_phase_loss


def train_joint_model(base, wrapper, model, x_norm, y_norm, masks, sim, y_mean, y_std, device):
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["val"])[0]
    train_ds = TensorDataset(
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(x_norm[train_idx], dtype=base.REAL_DTYPE),
        torch.tensor(y_norm[train_idx], dtype=base.REAL_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    omega_t = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    ri_scale = target_ri_scale(base, sim.target_s, masks["train"], device)

    val_x = torch.tensor(x_norm[val_idx], dtype=base.REAL_DTYPE, device=device)
    val_y = torch.tensor(y_norm[val_idx], dtype=base.REAL_DTYPE, device=device)
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
        total_param = 0.0
        total_ri = 0.0
        total_mag = 0.0
        seen = 0
        for idx_b, xb, yb in loader:
            idx_np = idx_b.numpy()
            xb = xb.to(device)
            yb = yb.to(device)
            base_b = torch.tensor(sim.base_abcds[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            target_b = torch.tensor(sim.target_s[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm, _, pred_s = cascade_shared_model(base, wrapper, model, xb, base_b, omega_t, y_mean_t, y_std_t)
            loss, param_loss, ri_loss, mag_loss = joint_loss(base, pred_norm, yb, pred_s, target_b, ri_scale)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            batch_n = len(xb)
            total_loss += float(loss.detach().cpu()) * batch_n
            total_param += float(param_loss.detach().cpu()) * batch_n
            total_ri += float(ri_loss.detach().cpu()) * batch_n
            total_mag += float(mag_loss.detach().cpu()) * batch_n
            seen += batch_n

        model.eval()
        with torch.no_grad():
            val_norm, _, val_s = cascade_shared_model(base, wrapper, model, val_x, val_base, omega_t, y_mean_t, y_std_t)
            val_loss, val_param, val_ri, val_mag = joint_loss(base, val_norm, val_y, val_s, val_target, ri_scale)
        train_loss = total_loss / max(seen, 1)
        row = {
            "stage": "joint_param_sparam",
            "epoch": epoch,
            "train_total_loss": train_loss,
            "train_param_loss": total_param / max(seen, 1),
            "train_ri_loss": total_ri / max(seen, 1),
            "train_mag_phase_loss": total_mag / max(seen, 1),
            "val_total_loss": float(val_loss.detach().cpu()),
            "val_param_loss": float(val_param.detach().cpu()),
            "val_ri_loss": float(val_ri.detach().cpu()),
            "val_mag_phase_loss": float(val_mag.detach().cpu()),
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
                f"[joint] epoch={epoch}, train={row['train_total_loss']:.4e}, "
                f"val={row['val_total_loss']:.4e}, val_ri={row['val_ri_loss']:.4e}, "
                f"val_param={row['val_param_loss']:.4e}",
                flush=True,
            )
        if stale >= JOINT_PATIENCE:
            break

    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def parameter_sign_stats(table: pd.DataFrame, value_prefix: str, param_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in param_names:
        col = f"{value_prefix}_{name}" if value_prefix else name
        values = table[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "value_set": value_prefix or "target",
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


def save_joint_plots(base, output_dir: Path, param_history: pd.DataFrame, joint_history: pd.DataFrame, pre_summary: pd.DataFrame, final_summary: pd.DataFrame):
    fig, axes = base.plt.subplots(1, 2, figsize=(13, 4), dpi=150)
    axes[0].plot(param_history["epoch"], param_history["train_loss"], label="param train")
    axes[0].plot(param_history["epoch"], param_history["val_loss"], label="param val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalized parameter MSE")
    axes[0].set_title("Parameter pretrain")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(joint_history["epoch"], joint_history["train_total_loss"], label="joint train")
    axes[1].plot(joint_history["epoch"], joint_history["val_total_loss"], label="joint val")
    axes[1].plot(joint_history["epoch"], joint_history["val_ri_loss"], label="val RI")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Joint fine-tune")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "joint_training_loss.png")
    base.plt.close(fig)

    merged = pre_summary[["split", "nn_nmse_mean"]].rename(columns={"nn_nmse_mean": "param_only_nn"}).merge(
        final_summary[["split", "nn_nmse_mean"]].rename(columns={"nn_nmse_mean": "joint_nn"}),
        on="split",
        how="inner",
    )
    fig, ax = base.plt.subplots(figsize=(7, 4), dpi=150)
    x = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width / 2, merged["param_only_nn"], width, label="param only NN")
    ax.bar(x + width / 2, merged["joint_nn"], width, label="joint NN")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["split"])
    ax.set_ylabel("NMSE mean")
    ax.set_title("NN cascade NMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "param_only_vs_joint_nmse.png")
    base.plt.close(fig)


def main() -> None:
    nnsrc = load_module(NN_SOURCE_SCRIPT, "v11_param_nn_source_for_joint")
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_joint")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_wrapper_for_joint")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_joint")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    opt_dir = version_root / "results" / OPT_RESULT_LABEL
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
    opt_targets = pd.read_csv(opt_target_path, encoding="utf-8-sig")
    dut_df = wrapper.collect_v11_samples(base)
    if list(dut_df["sample_id"]) != list(opt_targets["sample_id"]):
        opt_targets = dut_df[["sample_id"]].merge(opt_targets, on="sample_id", how="left")
    if opt_targets[wrapper.V08_PARAM_NAMES].isna().any().any():
        raise ValueError("Optimized target table is missing parameter values after sample alignment.")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    sim = base.load_single_device_simulation(dut_df, settings)
    masks = nnsrc.split_masks(opt_targets)
    x_raw = opt_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = opt_targets[wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = nnsrc.normalize_by_train(x_raw, masks["train"])
    y_norm, y_mean, y_std = nnsrc.normalize_by_train(y_raw, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    model = wrapper.SharedV08ParamNet(input_dim=x_norm.shape[1]).to(dtype=base.REAL_DTYPE, device=device)

    param_history = nnsrc.train_param_model(base, wrapper, model, x_norm, y_norm, masks, device)
    pre_pred_params = nnsrc.predict_params(base, model, x_norm, y_mean, y_std, device)
    pre_metrics, pre_pred_table = nnsrc.evaluate_predictions(base, wrapper, dut_df, sim, opt_targets, pre_pred_params)
    pre_summary = nnsrc.summarize(pre_metrics)

    joint_history = train_joint_model(base, wrapper, model, x_norm, y_norm, masks, sim, y_mean, y_std, device)
    final_pred_params = nnsrc.predict_params(base, model, x_norm, y_mean, y_std, device)
    final_metrics, final_pred_table = nnsrc.evaluate_predictions(base, wrapper, dut_df, sim, opt_targets, final_pred_params)
    final_summary = nnsrc.summarize(final_metrics)

    target_sign_stats = parameter_sign_stats(opt_targets, "", wrapper.V08_PARAM_NAMES)
    final_sign_stats = parameter_sign_stats(final_pred_table, "pred", wrapper.V08_PARAM_NAMES)
    sign_stats = pd.concat([target_sign_stats, final_sign_stats], ignore_index=True)

    nnsrc.save_summary_plots(base, output_dir, param_history, final_metrics, final_pred_table, wrapper)
    plot_dir, plot_paths = nnsrc.save_comparison_plots(
        base,
        wrapper,
        output_dir,
        dut_df,
        sim,
        final_metrics,
        opt_targets,
        final_pred_params,
    )
    save_joint_plots(base, output_dir, param_history, joint_history, pre_summary, final_summary)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": "v11_shared7_param_sparam_joint_adslen09",
                "architecture": "seven independent input->30->30->20->1 networks",
                "feature_columns": base.STRUCTURE_COLUMNS,
                "target_columns": wrapper.V08_PARAM_NAMES,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "source_optimized_targets": str(opt_target_path),
                "source_ads_cache": str(base.ADS_CACHE_DIR),
                "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
                "joint_loss": {
                    "ri_loss_weight": 1.0,
                    "param_anchor_weight": PARAM_ANCHOR_WEIGHT,
                    "mag_phase_weight": MAG_PHASE_WEIGHT,
                },
                "connection_count": wrapper.CONNECTION_COUNT,
                "v08_scale_factors": wrapper.V08_SCALE_FACTORS.tolist(),
                "freq_hz": sim.freq_hz.tolist(),
            },
        },
        output_dir / "shared7_param_sparam_joint.pt",
    )

    param_history.to_csv(output_dir / "param_pretrain_history.csv", index=False, encoding="utf-8-sig")
    joint_history.to_csv(output_dir / "joint_training_history.csv", index=False, encoding="utf-8-sig")
    pre_metrics.to_csv(output_dir / "param_only_metrics.csv", index=False, encoding="utf-8-sig")
    pre_summary.to_csv(output_dir / "param_only_summary.csv", index=False, encoding="utf-8-sig")
    pre_pred_table.to_csv(output_dir / "param_only_predictions.csv", index=False, encoding="utf-8-sig")
    final_metrics.to_csv(output_dir / "joint_metrics.csv", index=False, encoding="utf-8-sig")
    final_summary.to_csv(output_dir / "joint_summary.csv", index=False, encoding="utf-8-sig")
    final_pred_table.to_csv(output_dir / "joint_predictions.csv", index=False, encoding="utf-8-sig")
    sign_stats.to_csv(output_dir / "target_and_joint_predicted_parameter_sign_stats.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "source_optimized_targets": str(opt_target_path),
        "source_ads_cache": str(base.ADS_CACHE_DIR),
        "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
        "architecture": "seven independent input->30->30->20->1 networks",
        "loss": {
            "total": "s11_s21_ri_loss + param_anchor_weight * normalized_parameter_mse + mag_phase_weight * mag_phase_loss",
            "param_anchor_weight": PARAM_ANCHOR_WEIGHT,
            "mag_phase_weight": MAG_PHASE_WEIGHT,
        },
        "samples": int(len(dut_df)),
        "param_epochs_completed": int(param_history["epoch"].iloc[-1]) if len(param_history) else 0,
        "joint_epochs_completed": int(joint_history["epoch"].iloc[-1]) if len(joint_history) else 0,
        "param_only_summary": pre_summary.to_dict(orient="records"),
        "joint_summary": final_summary.to_dict(orient="records"),
        "parameter_sign_stats": sign_stats.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_paths,
    }
    (output_dir / "joint_training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "joint_training_report.md").write_text(
        "\n".join(
            [
                "# V11 ADS Length 0.9 Joint Parameter and S-Parameter Training",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source optimized targets: `{opt_target_path}`",
                f"- Source ADS cache: `{base.ADS_CACHE_DIR}`",
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}`",
                "- Network: seven independent `input -> 30 -> 30 -> 20 -> 1` models.",
                f"- Loss: `RI S-parameter loss + {PARAM_ANCHOR_WEIGHT} * parameter MSE + {MAG_PHASE_WEIGHT} * mag/phase loss`.",
                f"- Parameter pretrain epochs: `{report['param_epochs_completed']}`",
                f"- Joint epochs: `{report['joint_epochs_completed']}`",
                "",
                "## Param-Only Summary",
                "",
                dataframe_to_markdown(pre_summary),
                "",
                "## Joint Summary",
                "",
                dataframe_to_markdown(final_summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
                "",
                "## Outputs",
                "",
                f"- Checkpoint: `{output_dir / 'shared7_param_sparam_joint.pt'}`",
                f"- Joint metrics: `{output_dir / 'joint_metrics.csv'}`",
                f"- Joint summary: `{output_dir / 'joint_summary.csv'}`",
                f"- Joint predictions: `{output_dir / 'joint_predictions.csv'}`",
                f"- Parameter sign stats: `{output_dir / 'target_and_joint_predicted_parameter_sign_stats.csv'}`",
                f"- Training plot: `{output_dir / 'joint_training_loss.png'}`",
                f"- NMSE comparison plot: `{output_dir / 'param_only_vs_joint_nmse.png'}`",
                f"- S-parameter comparison plots: `{plot_dir}`",
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
                f"- Source ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Parameter pretrain epochs: `{report['param_epochs_completed']}`",
                f"- Joint epochs: `{report['joint_epochs_completed']}`",
                f"- Comparison plots: `{len(plot_paths)}`",
                "",
                "## Param-Only Summary",
                "",
                dataframe_to_markdown(pre_summary),
                "",
                "## Joint Summary",
                "",
                dataframe_to_markdown(final_summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
            ]
        ),
        encoding="utf-8",
    )
    print("Param-only summary:", flush=True)
    print(dataframe_to_markdown(pre_summary), flush=True)
    print("Joint summary:", flush=True)
    print(dataframe_to_markdown(final_summary), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
