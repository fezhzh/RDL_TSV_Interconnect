# -*- coding: utf-8 -*-
"""Diagnose resonance-like spikes in the v11 S-parameter-only multi-head result.

Run this file directly in VS Code. No command-line arguments are required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_v11_multihead_exclude_resonance_sparam_only_adslen09.py"
RUN_LABEL = "v11_multihead_exclude_resonance_sparam_only_adslen09"


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


def main() -> None:
    train = load_module(TRAIN_SCRIPT, "v11_sparam_only_train_for_resonance_diag")
    source = train.load_module(train.SOURCE_SCRIPT, "v11_resonance_diag_source")
    wrapper = train.load_module(train.WRAPPER_SCRIPT, "v11_resonance_diag_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_resonance_diag_base")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    output_dir = version_root / "results" / RUN_LABEL
    opt_dir = version_root / "results" / train.OPT_RESULT_LABEL
    source_ads_dir = version_root / "results" / train.SOURCE_ADS_LABEL

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = source_ads_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = train.ADS_DEVICE_LENGTH_SCALE

    dut_all = wrapper.collect_v11_samples(base)
    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = train.ADS_DEVICE_LENGTH_SCALE
    sim_all = base.load_single_device_simulation(dut_all, settings)

    resonance_filter = pd.read_csv(output_dir / "resonance_diagnostics_all_samples.csv", encoding="utf-8-sig")
    active_ids = set(resonance_filter.loc[~resonance_filter["is_resonant"], "sample_id"])
    dut_df = dut_all[dut_all["sample_id"].isin(active_ids)].reset_index(drop=True)
    source_index = [int(i) for i in dut_all.index[dut_all["sample_id"].isin(active_ids)]]
    base_abcds = sim_all.base_abcds[source_index]

    pred = pd.read_csv(output_dir / "multihead_joint_predictions.csv", encoding="utf-8-sig").set_index("sample_id")
    metrics = pd.read_csv(output_dir / "multihead_joint_metrics.csv", encoding="utf-8-sig")
    opt_targets = pd.read_csv(opt_dir / train.OPT_TARGET_FILE, encoding="utf-8-sig").set_index("sample_id")
    columns = train.multihead_target_columns(wrapper)
    omega = 2.0 * np.pi * sim_all.freq_hz

    rows = []
    for i, row in dut_df.iterrows():
        sample_id = str(row["sample_id"])
        pred_params = pred.loc[sample_id, [f"pred_{col}" for col in columns]].to_numpy(dtype=np.float64)
        pred_s = base.abcd2s(wrapper.cascade_with_v08_np(base, base_abcds[i], omega, pred_params))
        opt_params = opt_targets.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        opt_s = base.abcd2s(wrapper.cascade_with_v08_np(base, base_abcds[i], omega, opt_params))

        pred_ri = train.s11_s21_ri_np(pred_s)
        opt_ri = train.s11_s21_ri_np(opt_s)
        pred_db = np.column_stack([train.db20(pred_s[:, 0, 0]), train.db20(pred_s[:, 1, 0])])
        opt_db = np.column_stack([train.db20(opt_s[:, 0, 0]), train.db20(opt_s[:, 1, 0])])
        rows.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "pred_max_d1": float(np.max(np.abs(np.diff(pred_ri, axis=0)))),
                "pred_db_d1": float(np.max(np.abs(np.diff(pred_db, axis=0)))),
                "opt_max_d1": float(np.max(np.abs(np.diff(opt_ri, axis=0)))),
                "opt_db_d1": float(np.max(np.abs(np.diff(opt_db, axis=0)))),
            }
        )

    diag = pd.DataFrame(rows).merge(
        metrics[["sample_id", "nn_nmse_s11_s21_ri", "optimized_nmse_s11_s21_ri"]],
        on="sample_id",
        how="left",
    )
    diag["pred_is_resonant"] = (diag["pred_db_d1"] > train.RESONANCE_DB_D1_THRESHOLD) | (
        diag["pred_max_d1"] > train.RESONANCE_RI_D1_THRESHOLD
    )
    diag.to_csv(output_dir / "sparam_only_prediction_resonance_diagnostics.csv", index=False, encoding="utf-8-sig")

    summary = (
        diag.groupby(["split", "pred_is_resonant"], as_index=False)
        .agg(
            count=("sample_id", "count"),
            pred_db_d1_mean=("pred_db_d1", "mean"),
            pred_max_d1_mean=("pred_max_d1", "mean"),
            nn_nmse_mean=("nn_nmse_s11_s21_ri", "mean"),
        )
    )
    summary.to_csv(output_dir / "sparam_only_prediction_resonance_summary.csv", index=False, encoding="utf-8-sig")
    worst = diag.sort_values(["pred_db_d1", "pred_max_d1"], ascending=False).head(20)
    worst.to_csv(output_dir / "sparam_only_worst_prediction_resonance_samples.csv", index=False, encoding="utf-8-sig")

    param_rows = []
    for name in wrapper.V08_PARAM_NAMES:
        vals = []
        for idx in range(1, wrapper.CONNECTION_COUNT + 1):
            vals.append(pred[f"pred_conn{idx}_{name}"].to_numpy(dtype=np.float64))
        values = np.concatenate(vals)
        param_rows.append(
            {
                "parameter": name,
                "min": float(np.min(values)),
                "p01": float(np.quantile(values, 0.01)),
                "median": float(np.quantile(values, 0.5)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
                "negative_count": int(np.sum(values < 0.0)),
                "total_count": int(len(values)),
            }
        )
    param_summary = pd.DataFrame(param_rows)
    param_summary.to_csv(output_dir / "sparam_only_predicted_parameter_range_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "current_prediction_resonant_count": int(diag["pred_is_resonant"].sum()),
        "active_sample_count": int(len(diag)),
        "summary": summary.to_dict(orient="records"),
        "worst_samples": worst.to_dict(orient="records"),
        "parameter_summary": param_summary.to_dict(orient="records"),
    }
    (output_dir / "sparam_only_resonance_diagnosis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "sparam_only_resonance_diagnosis_report.md").write_text(
        "\n".join(
            [
                "# S-Parameter-Only Multi-Head Resonance Diagnosis",
                "",
                f"- Output: `{output_dir}`",
                f"- Active samples: `{len(diag)}`",
                f"- Current prediction resonant samples: `{int(diag['pred_is_resonant'].sum())}`",
                "",
                "## Current Prediction Resonance Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Worst Current Prediction Resonance Samples",
                "",
                dataframe_to_markdown(worst[["sample_id", "split", "pred_db_d1", "pred_max_d1", "nn_nmse_s11_s21_ri"]]),
                "",
                "## Predicted Parameter Range Summary",
                "",
                dataframe_to_markdown(param_summary),
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)
    print(dataframe_to_markdown(worst[["sample_id", "split", "pred_db_d1", "pred_max_d1", "nn_nmse_s11_s21_ri"]]), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
