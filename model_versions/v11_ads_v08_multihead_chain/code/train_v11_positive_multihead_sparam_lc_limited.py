# -*- coding: utf-8 -*-
"""Train positive multi-head NN with S-parameter loss plus L/C limit penalty.

Run this file directly in VS Code. No command-line arguments are required.

Physical limits:
    Cn1/Cn2/Cn3 < 1e-11 F
    Ln1 < 1e-8 H

With the current v08 scale factors this is equivalent to:
    Cn*_scale < 1000
    Ln1_scale < 1000
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
BASE_TRAIN_SCRIPT = THIS_DIR / "train_v11_positive_multihead_sparam_from_shared.py"

RUN_LABEL = "v11_positive_multihead_sparam_lc_limited_log_adslen09"
C_LIMIT_F = 1e-11
L_LIMIT_H = 1e-8
LC_PENALTY_WEIGHT = 0.25
JOINT_EPOCHS = 320
JOINT_PATIENCE = 55
JOINT_LR = 2e-5
BATCH_SIZE = 8
PRINT_EVERY = 10


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
        vals = []
        for col in columns:
            value = row[col]
            vals.append(f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def lc_limit_loss(base, wrapper, p_all):
    scales = torch.tensor(wrapper.V08_SCALE_FACTORS, dtype=base.REAL_DTYPE, device=p_all.device)
    physical = p_all * scales
    c_vals = physical[..., [0, 2, 4]]
    l_vals = physical[..., [6]]
    c_penalty = torch.mean(torch.relu(c_vals / C_LIMIT_F - 1.0) ** 2)
    l_penalty = torch.mean(torch.relu(l_vals / L_LIMIT_H - 1.0) ** 2)
    return c_penalty + l_penalty, c_penalty, l_penalty


def train_sparam_lc_limited(base_train, base, wrapper, model, x_norm, masks, sim, y_mean, y_std, device):
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["val"])[0]
    train_ds = TensorDataset(torch.tensor(train_idx, dtype=torch.long), torch.tensor(x_norm[train_idx], dtype=base.REAL_DTYPE))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    omega_t = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    ri_scale = base_train.target_ri_scale(base, sim.target_s, masks["train"], device)
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
        total_s = 0.0
        total_lc = 0.0
        total = 0.0
        seen = 0
        for idx_b, xb in loader:
            idx_np = idx_b.numpy()
            xb = xb.to(device)
            base_b = torch.tensor(sim.base_abcds[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            target_b = torch.tensor(sim.target_s[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            optimizer.zero_grad(set_to_none=True)
            _, p_all, _ = base_train.denorm_log_to_positive_params(
                base,
                model(xb),
                y_mean_t,
                y_std_t,
                wrapper.CONNECTION_COUNT,
                len(wrapper.V08_PARAM_NAMES),
            )
            pred_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t))
            s_loss = base_train.multihead_sparam_loss(pred_s, target_b, ri_scale)
            lc_loss, _, _ = lc_limit_loss(base, wrapper, p_all)
            loss = s_loss + LC_PENALTY_WEIGHT * lc_loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(xb)
            total_s += float(s_loss.detach().cpu()) * n
            total_lc += float(lc_loss.detach().cpu()) * n
            total += float(loss.detach().cpu()) * n
            seen += n

        model.eval()
        with torch.no_grad():
            _, val_all, _ = base_train.denorm_log_to_positive_params(
                base,
                model(val_x),
                y_mean_t,
                y_std_t,
                wrapper.CONNECTION_COUNT,
                len(wrapper.V08_PARAM_NAMES),
            )
            val_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, val_base, val_all, omega_t))
            val_s_loss = base_train.multihead_sparam_loss(val_s, val_target, ri_scale)
            val_lc_loss, val_c_loss, val_l_loss = lc_limit_loss(base, wrapper, val_all)
            val_loss = val_s_loss + LC_PENALTY_WEIGHT * val_lc_loss

        row = {
            "stage": "multihead_sparam_lc_limited",
            "epoch": epoch,
            "train_total_loss": float(total / max(seen, 1)),
            "train_ri_loss": float(total_s / max(seen, 1)),
            "train_lc_loss": float(total_lc / max(seen, 1)),
            "val_total_loss": float(val_loss.detach().cpu()),
            "val_ri_loss": float(val_s_loss.detach().cpu()),
            "val_lc_loss": float(val_lc_loss.detach().cpu()),
            "val_c_loss": float(val_c_loss.detach().cpu()),
            "val_l_loss": float(val_l_loss.detach().cpu()),
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
                f"[multihead-lc] epoch={epoch}, train_ri={row['train_ri_loss']:.4e}, "
                f"train_lc={row['train_lc_loss']:.4e}, val_ri={row['val_ri_loss']:.4e}, "
                f"val_lc={row['val_lc_loss']:.4e}",
                flush=True,
            )
        if stale >= JOINT_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def lc_limit_stats(pred_table: pd.DataFrame, wrapper) -> pd.DataFrame:
    rows = []
    for name in ["Cn1_scale", "Cn2_scale", "Cn3_scale", "Ln1_scale"]:
        cols = [f"pred_conn{idx}_{name}" for idx in range(1, wrapper.CONNECTION_COUNT + 1)]
        values = pred_table[cols].to_numpy(dtype=np.float64).ravel()
        scale = 1e-14 if name.startswith("Cn") else 1e-11
        physical = values * scale
        limit = C_LIMIT_F if name.startswith("Cn") else L_LIMIT_H
        rows.append(
            {
                "parameter": name,
                "scale_limit": float(limit / scale),
                "physical_limit": limit,
                "scale_min": float(np.min(values)),
                "scale_p95": float(np.quantile(values, 0.95)),
                "scale_p99": float(np.quantile(values, 0.99)),
                "scale_max": float(np.max(values)),
                "physical_max": float(np.max(physical)),
                "exceed_count": int(np.sum(physical > limit)),
                "total_count": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    base_train = load_module(BASE_TRAIN_SCRIPT, "v11_positive_multihead_base_train_lc")
    source = base_train.load_module(base_train.SOURCE_SCRIPT, "v11_lc_source")
    positive = base_train.load_module(base_train.POSITIVE_SCRIPT, "v11_lc_positive")
    wrapper = base_train.load_module(base_train.WRAPPER_SCRIPT, "v11_lc_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_lc_base")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    opt_dir = version_root / "results" / base_train.OPT_RESULT_LABEL
    shared_nn_dir = version_root / "results" / base_train.SHARED_NN_LABEL
    source_ads_dir = version_root / "results" / base_train.SOURCE_ADS_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = source_ads_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = base_train.ADS_DEVICE_LENGTH_SCALE
    base.set_seed(base.RANDOM_SEED)

    opt_targets_all = pd.read_csv(opt_dir / base_train.OPT_TARGET_FILE, encoding="utf-8-sig")
    dut_all = positive.collect_lhs400_rdl_tsv_samples(base)
    target_ids = set(opt_targets_all["sample_id"].astype(str))
    excluded_unoptimized = dut_all[~dut_all["sample_id"].astype(str).isin(target_ids)].copy()
    dut_df = dut_all[dut_all["sample_id"].astype(str).isin(target_ids)].reset_index(drop=True)
    opt_targets = dut_df[["sample_id"]].merge(opt_targets_all, on="sample_id", how="left")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = base_train.ADS_DEVICE_LENGTH_SCALE
    settings["ads_geometry_scale_note"] = "Applied to l_tmrdl, l_bsmrdl, and h_tsv by the v11 base ADS runner."
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)

    masks = base_train.split_masks(opt_targets)
    x_raw = opt_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_linear = opt_targets[wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    y_log_shared = np.log10(np.clip(y_linear, base_train.POSITIVE_LOWER, base_train.POSITIVE_UPPER))
    y_log_multi = base_train.repeat_shared_targets(y_log_shared, wrapper)
    x_norm, x_mean, x_std = base_train.normalize_by_train(x_raw, masks["train"])
    y_log_multi_norm, y_log_multi_mean, y_log_multi_std = base_train.normalize_by_train(y_log_multi, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    shared_ckpt_path = shared_nn_dir / "positive_shared7_param_nns_log.pt"
    _, model, shared_checkpoint = base_train.load_initialized_models(base, wrapper, shared_ckpt_path, device)

    initial_metrics, initial_pred = base_train.evaluate_multihead(base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_log_multi_mean, y_log_multi_std, device)
    initial_summary = base_train.summarize(initial_metrics)
    history = train_sparam_lc_limited(base_train, base, wrapper, model, x_norm, masks, sim, y_log_multi_mean, y_log_multi_std, device)
    metrics, pred_table = base_train.evaluate_multihead(base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_log_multi_mean, y_log_multi_std, device)
    summary = base_train.summarize(metrics)
    sign_stats = base_train.parameter_sign_stats(pred_table, wrapper)
    lc_stats = lc_limit_stats(pred_table, wrapper)
    plot_dir, plot_paths = base_train.save_plots(base, wrapper, output_dir, history.rename(columns={"train_total_loss": "train_ri_loss", "val_total_loss": "val_ri_loss"}), metrics, dut_df, sim, opt_targets, pred_table)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": RUN_LABEL,
                "architecture": "positive multi-head initialized from shared 30-30-20 networks",
                "training_objective": "S11/S21 real/imag loss plus L/C physical limit penalty",
                "lc_limits": {"C_limit_F": C_LIMIT_F, "L_limit_H": L_LIMIT_H, "penalty_weight": LC_PENALTY_WEIGHT},
                "equivalent_scale_limits": {"Cn_scale_limit": C_LIMIT_F / 1e-14, "Ln1_scale_limit": L_LIMIT_H / 1e-11},
                "feature_columns": base.STRUCTURE_COLUMNS,
                "target_columns": base_train.multihead_target_columns(wrapper),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_log_multi_mean": y_log_multi_mean.tolist(),
                "y_log_multi_std": y_log_multi_std.tolist(),
                "source_shared_checkpoint": str(shared_ckpt_path),
                "source_optimized_targets": str(opt_dir / base_train.OPT_TARGET_FILE),
                "ads_device_length_scale": base_train.ADS_DEVICE_LENGTH_SCALE,
                "excluded_unoptimized_sample_ids": excluded_unoptimized["sample_id"].astype(str).tolist(),
                "shared_checkpoint_metadata": shared_checkpoint.get("metadata", {}),
            },
        },
        output_dir / "positive_multihead_sparam_lc_limited.pt",
    )

    history.to_csv(output_dir / "positive_multihead_sparam_lc_limited_history.csv", index=False, encoding="utf-8-sig")
    initial_metrics.to_csv(output_dir / "initial_shared_expanded_metrics.csv", index=False, encoding="utf-8-sig")
    initial_summary.to_csv(output_dir / "initial_shared_expanded_summary.csv", index=False, encoding="utf-8-sig")
    initial_pred.to_csv(output_dir / "initial_shared_expanded_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "positive_multihead_sparam_lc_limited_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "positive_multihead_sparam_lc_limited_summary.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(output_dir / "positive_multihead_sparam_lc_limited_predictions.csv", index=False, encoding="utf-8-sig")
    sign_stats.to_csv(output_dir / "positive_multihead_lc_limited_parameter_sign_stats.csv", index=False, encoding="utf-8-sig")
    lc_stats.to_csv(output_dir / "positive_multihead_lc_limit_stats.csv", index=False, encoding="utf-8-sig")
    excluded_unoptimized.to_csv(output_dir / "excluded_unoptimized_samples.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "samples": int(len(dut_df)),
        "excluded_unoptimized_samples": int(len(excluded_unoptimized)),
        "source_shared_checkpoint": str(shared_ckpt_path),
        "lc_limits": {"C_limit_F": C_LIMIT_F, "L_limit_H": L_LIMIT_H, "penalty_weight": LC_PENALTY_WEIGHT},
        "equivalent_scale_limits": {"Cn_scale_limit": C_LIMIT_F / 1e-14, "Ln1_scale_limit": L_LIMIT_H / 1e-11},
        "epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
        "best_val_total_loss": float(history["val_total_loss"].min()) if len(history) else None,
        "initial_summary": initial_summary.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "lc_stats": lc_stats.to_dict(orient="records"),
        "parameter_sign_stats": sign_stats.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_paths,
    }
    (output_dir / "positive_multihead_sparam_lc_limited_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "positive_multihead_sparam_lc_limited_report.md").write_text(
        "\n".join(
            [
                "# V11 Positive Multi-Head S-Parameter Training With L/C Limit Penalty",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source shared checkpoint: `{shared_ckpt_path}`",
                "- Training target: cascaded `S11/S21` real/imag loss plus L/C physical-limit penalty.",
                f"- C limit: `{C_LIMIT_F}` F, equivalent to `Cn*_scale < {C_LIMIT_F / 1e-14:.6g}`.",
                f"- L limit: `{L_LIMIT_H}` H, equivalent to `Ln1_scale < {L_LIMIT_H / 1e-11:.6g}`.",
                f"- L/C penalty weight: `{LC_PENALTY_WEIGHT}`",
                f"- Epochs completed: `{report['epochs_completed']}`",
                "",
                "## Reasonableness Analysis",
                "",
                "- The C limit is reasonable as a stabilizing constraint for the current model because the previous resonance diagnosis showed `Cn3_scale` p99 `11626.8` and max `84489.8`, which correspond to physical capacitances far above `1e-11 F`.",
                "- The L limit is physically permissive in this parameterization. Since `Ln1_scale * 1e-11 = L(H)`, `L < 1e-8 H` means `Ln1_scale < 1000`; the previous NN max `Ln1_scale` was only about `6.8`, so this loss term will almost never activate.",
                "- Because the optimized target table itself has `Cn3_scale` max `3523` (`3.523e-11 F`), the C limit can conflict with a few optimized fitting targets. This is acceptable if the priority is suppressing resonance, but it may raise the best achievable S-parameter error for those samples.",
                "",
                "## Initial Shared-Expanded Summary",
                "",
                dataframe_to_markdown(initial_summary),
                "",
                "## L/C-Limited Multi-Head Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## L/C Limit Stats",
                "",
                dataframe_to_markdown(lc_stats),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
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
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- C limit: `{C_LIMIT_F}` F",
                f"- L limit: `{L_LIMIT_H}` H",
                f"- L/C penalty weight: `{LC_PENALTY_WEIGHT}`",
                f"- Comparison plots: `{len(plot_paths)}`",
                "",
                "## L/C-Limited Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## L/C Limit Stats",
                "",
                dataframe_to_markdown(lc_stats),
            ]
        ),
        encoding="utf-8",
    )
    print("L/C-limited summary:", flush=True)
    print(dataframe_to_markdown(summary), flush=True)
    print("L/C limit stats:", flush=True)
    print(dataframe_to_markdown(lc_stats), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
