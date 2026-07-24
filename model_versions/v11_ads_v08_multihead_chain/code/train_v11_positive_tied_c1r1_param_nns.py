# -*- coding: utf-8 -*-
"""Train 30-30-20 parameter NNs from tied Cn1/Cn2 Rn1/Rn2 targets.

Run this file directly in VS Code. No command-line arguments are required.

Five independent networks are trained for the tied-topology free parameters:
``Cn1_scale, Rn1_scale, Cn3_scale, Rn3_scale, Ln1_scale``. Predictions are
expanded to the seven-parameter circuit as ``Cn2=Cn1`` and ``Rn2=Rn1`` before
cascade evaluation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
NN_SOURCE_SCRIPT = THIS_DIR / "train_v11_shared7_param_nns_from_all_goodstart.py"
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
POSITIVE_SCRIPT = THIS_DIR / "optimize_v11_positive_shared_connection_lhs400_adslen09.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"

OPT_RESULT_LABEL = "v11_positive_tied_c1r1_sharedopt_lhs400_connection2_goodstart_all_adslen09"
OPT_TARGET_FILE = "v08_positive_goodstart_targets.csv"
SOURCE_ADS_LABEL = "v11_positive_sharedopt_lhs400_connection2_adslen09"
RUN_LABEL = "v11_positive_tied_c1r1_param_nns_log_adslen09"

ADS_DEVICE_LENGTH_SCALE = 0.9
POSITIVE_LOWER = 1e-9
POSITIVE_UPPER = 1e5
PLOT_WORST_VAL = 12
PLOT_WORST_ALL = 12
SIGN_EPS = 1e-12
TIED_PARAM_NAMES = ["Cn1_scale", "Rn1_scale", "Cn3_scale", "Rn3_scale", "Ln1_scale"]


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


def expand_tied_params(p5: np.ndarray) -> np.ndarray:
    p5 = np.asarray(p5, dtype=np.float64)
    return np.column_stack([p5[:, 0], p5[:, 1], p5[:, 0], p5[:, 1], p5[:, 2], p5[:, 3], p5[:, 4]])


def make_tied_model(input_dim: int, output_dim: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, output_dim),
    )


def add_log_prediction_columns(pred_table: pd.DataFrame, log_targets: np.ndarray, log_preds: np.ndarray) -> pd.DataFrame:
    out = pred_table.copy()
    for i, name in enumerate(TIED_PARAM_NAMES):
        out[f"target_log10_{name}"] = log_targets[:, i]
        out[f"pred_log10_{name}"] = log_preds[:, i]
        out[f"abs_log10_error_{name}"] = np.abs(log_preds[:, i] - log_targets[:, i])
    return out


def save_selected_comparison_plots(base, wrapper, nnsrc, output_dir: Path, dut_df: pd.DataFrame, sim, metrics: pd.DataFrame, opt_targets: pd.DataFrame, pred_params: np.ndarray):
    plot_dir = output_dir / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.concat(
        [
            metrics[metrics["split"].eq("val")].sort_values("nn_nmse_s11_s21_ri", ascending=False).head(PLOT_WORST_VAL),
            metrics.sort_values("nn_nmse_s11_s21_ri", ascending=False).head(PLOT_WORST_ALL),
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")
    paths = []
    for _, metric in selected.iterrows():
        idx = int(dut_df.index[dut_df["sample_id"].eq(metric["sample_id"])][0])
        out_path = plot_dir / f"{metric['sample_id']}.png"
        nnsrc.plot_comparison(base, wrapper, sim, idx, metric, opt_targets, pred_params, out_path)
        paths.append(str(out_path))
    return plot_dir, paths


def main() -> None:
    nnsrc = load_module(NN_SOURCE_SCRIPT, "v11_positive_param_nn_source")
    source = load_module(SOURCE_SCRIPT, "v11_positive_param_nn_calibrated_source")
    positive = load_module(POSITIVE_SCRIPT, "v11_positive_param_nn_positive_source")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_positive_param_nn_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_positive_param_nn_base")

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

    wrapper.V08_LOWER = np.full(len(wrapper.V08_PARAM_NAMES), POSITIVE_LOWER, dtype=np.float64)
    wrapper.V08_UPPER = np.full(len(wrapper.V08_PARAM_NAMES), POSITIVE_UPPER, dtype=np.float64)
    wrapper.V08_P0 = np.ones(len(wrapper.V08_PARAM_NAMES), dtype=np.float64)

    opt_target_path = opt_dir / OPT_TARGET_FILE
    opt_targets = pd.read_csv(opt_target_path, encoding="utf-8-sig")
    dut_all = positive.collect_lhs400_rdl_tsv_samples(base)
    target_ids = set(opt_targets["sample_id"].astype(str))
    excluded_unoptimized = dut_all[~dut_all["sample_id"].astype(str).isin(target_ids)].copy()
    dut_df = dut_all[dut_all["sample_id"].astype(str).isin(target_ids)].reset_index(drop=True)
    opt_targets = dut_df[["sample_id"]].merge(opt_targets, on="sample_id", how="left")
    if opt_targets[wrapper.V08_PARAM_NAMES].isna().any().any():
        raise ValueError("Tied optimized target table is missing parameter values after sample alignment.")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    settings["ads_geometry_scale_note"] = "Applied to l_tmrdl, l_bsmrdl, and h_tsv by the v11 base ADS runner."
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)

    masks = nnsrc.split_masks(opt_targets)
    x_raw = opt_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_linear = opt_targets[TIED_PARAM_NAMES].to_numpy(dtype=np.float64)
    y_clipped = np.clip(y_linear, POSITIVE_LOWER, POSITIVE_UPPER)
    y_log = np.log10(y_clipped)
    x_norm, x_mean, x_std = nnsrc.normalize_by_train(x_raw, masks["train"])
    y_log_norm, y_log_mean, y_log_std = nnsrc.normalize_by_train(y_log, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    model = make_tied_model(input_dim=x_norm.shape[1], output_dim=len(TIED_PARAM_NAMES)).to(dtype=base.REAL_DTYPE, device=device)
    history = nnsrc.train_param_model(base, wrapper, model, x_norm, y_log_norm, masks, device)
    pred_log = nnsrc.predict_params(base, model, x_norm, y_log_mean, y_log_std, device)
    pred_tied_params = np.clip(np.power(10.0, pred_log), POSITIVE_LOWER, POSITIVE_UPPER)
    pred_params = expand_tied_params(pred_tied_params)

    metrics, pred_table = nnsrc.evaluate_predictions(base, wrapper, dut_df, sim, opt_targets, pred_params)
    pred_table = add_log_prediction_columns(pred_table, y_log, pred_log)
    summary = nnsrc.summarize(metrics)
    target_sign_stats = parameter_sign_stats(opt_targets, "", wrapper.V08_PARAM_NAMES)
    pred_sign_stats = parameter_sign_stats(pred_table, "pred", wrapper.V08_PARAM_NAMES)
    sign_stats = pd.concat([target_sign_stats, pred_sign_stats], ignore_index=True)

    nnsrc.save_summary_plots(base, output_dir, history, metrics, pred_table, wrapper)
    plot_dir, plot_paths = save_selected_comparison_plots(base, wrapper, nnsrc, output_dir, dut_df, sim, metrics, opt_targets, pred_params)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "model_type": RUN_LABEL,
            "architecture": "five-output input->30->30->20 network for tied free parameters; expanded as Cn2=Cn1 and Rn2=Rn1",
            "target_transform": "log10 positive parameters; inverse power(10) clipped to [1e-9, 1e5]",
            "feature_columns": base.STRUCTURE_COLUMNS,
            "target_columns": TIED_PARAM_NAMES,
            "expanded_target_columns": wrapper.V08_PARAM_NAMES,
            "tied_topology": {"Cn2_scale": "Cn1_scale", "Rn2_scale": "Rn1_scale"},
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_log_mean": y_log_mean.tolist(),
            "y_log_std": y_log_std.tolist(),
            "positive_bounds": [POSITIVE_LOWER, POSITIVE_UPPER],
            "source_optimized_targets": str(opt_target_path),
            "source_ads_cache": str(base.ADS_CACHE_DIR),
            "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
            "excluded_unoptimized_sample_ids": excluded_unoptimized["sample_id"].astype(str).tolist(),
            "connection_count": wrapper.CONNECTION_COUNT,
            "v08_scale_factors": wrapper.V08_SCALE_FACTORS.tolist(),
            "freq_hz": sim.freq_hz.tolist(),
        },
    }
    torch.save(checkpoint, output_dir / "positive_tied_c1r1_param_nns_log.pt")

    history.to_csv(output_dir / "positive_tied_c1r1_param_training_history.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(output_dir / "positive_tied_c1r1_param_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "optimized_vs_positive_tied_c1r1_nn_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "optimized_vs_positive_tied_c1r1_nn_summary.csv", index=False, encoding="utf-8-sig")
    sign_stats.to_csv(output_dir / "target_and_predicted_parameter_sign_stats.csv", index=False, encoding="utf-8-sig")
    excluded_unoptimized.to_csv(output_dir / "excluded_unoptimized_samples.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "source_optimized_targets": str(opt_target_path),
        "source_ads_cache": str(base.ADS_CACHE_DIR),
        "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
        "architecture": "five-output input->30->30->20 network for tied free parameters",
        "target_transform": "log10 positive parameters; inverse clipped to positive bounds",
        "target_columns": TIED_PARAM_NAMES,
        "expanded_target_columns": wrapper.V08_PARAM_NAMES,
        "tied_topology": {"Cn2_scale": "Cn1_scale", "Rn2_scale": "Rn1_scale"},
        "samples": int(len(dut_df)),
        "excluded_unoptimized_samples": int(len(excluded_unoptimized)),
        "excluded_unoptimized_sample_ids": excluded_unoptimized["sample_id"].astype(str).tolist(),
        "epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
        "best_val_loss": float(history["val_loss"].min()) if len(history) else None,
        "summary": summary.to_dict(orient="records"),
        "parameter_sign_stats": sign_stats.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_paths,
    }
    (output_dir / "positive_tied_c1r1_param_nn_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "positive_tied_c1r1_param_nn_report.md").write_text(
        "\n".join(
            [
                "# V11 Positive Tied Cn1/Cn2 Rn1/Rn2 Parameter NN Report",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source optimized targets: `{opt_target_path}`",
                f"- Source ADS cache: `{base.ADS_CACHE_DIR}`",
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}`",
                f"- Samples used: `{len(dut_df)}`; unoptimized current-disk samples excluded: `{len(excluded_unoptimized)}`.",
                "- Network: `input -> 30 -> 30 -> 20 -> 5` for tied free parameters `Cn1/Rn1/Cn3/Rn3/Ln1`.",
                "- Expansion: `Cn2_scale=Cn1_scale` and `Rn2_scale=Rn1_scale` before cascade evaluation.",
                "- Target transform: train on `log10(parameter)` and convert back to positive scale clipped to `[1e-9, 1e5]`.",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Best validation loss: `{report['best_val_loss']}`",
                "",
                "## S-Parameter Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
                "",
                "## Outputs",
                "",
                f"- Checkpoint: `{output_dir / 'positive_tied_c1r1_param_nns_log.pt'}`",
                f"- Training history: `{output_dir / 'positive_tied_c1r1_param_training_history.csv'}`",
                f"- Parameter predictions: `{output_dir / 'positive_tied_c1r1_param_predictions.csv'}`",
                f"- Metrics: `{output_dir / 'optimized_vs_positive_tied_c1r1_nn_metrics.csv'}`",
                f"- Summary: `{output_dir / 'optimized_vs_positive_tied_c1r1_nn_summary.csv'}`",
                f"- Parameter sign stats: `{output_dir / 'target_and_predicted_parameter_sign_stats.csv'}`",
                f"- Excluded unoptimized samples: `{output_dir / 'excluded_unoptimized_samples.csv'}`",
                f"- Training/NMSE plot: `{output_dir / 'training_and_nmse_summary.png'}`",
                f"- Parameter scatter: `{output_dir / 'parameter_prediction_scatter.png'}`",
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
                f"- Network: `input -> 30 -> 30 -> 20 -> {len(TIED_PARAM_NAMES)}`",
                f"- Tied free parameters: `{', '.join(TIED_PARAM_NAMES)}`",
                "- Expansion: `Cn2_scale=Cn1_scale`, `Rn2_scale=Rn1_scale`",
                f"- Unoptimized current-disk samples excluded: `{len(excluded_unoptimized)}`",
                f"- Train/val/test evaluated: `{int(metrics['split'].eq('train').sum())}` / `{int(metrics['split'].eq('val').sum())}` / `{int(metrics['split'].eq('test').sum())}`",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Source ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Comparison plots: `{len(plot_paths)}`",
                f"- Predicted parameter nonpositive total: `{int(pred_sign_stats['negative_count'].sum() + pred_sign_stats['zero_count'].sum())}`",
                "",
                "## S-Parameter Summary",
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
    print(dataframe_to_markdown(summary), flush=True)
    print(dataframe_to_markdown(sign_stats), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
