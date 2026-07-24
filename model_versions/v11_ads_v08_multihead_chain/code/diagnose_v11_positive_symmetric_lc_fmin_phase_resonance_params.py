# -*- coding: utf-8 -*-
"""Diagnose resonance and large L/C outputs in the symmetric-LC fmin model.

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
TRAIN_SCRIPT = THIS_DIR / "train_v11_positive_multihead_sparam_from_shared.py"
RUN_LABEL = "v11_positive_symmetric_multihead_lc_fmin60_phase_log_adslen09"
DB_D1_THRESHOLD = 12.0
RI_D1_THRESHOLD = 0.2
CAP_PARAMS = ["Cn1_scale", "Cn2_scale", "Cn3_scale"]
IND_PARAMS = ["Ln1_scale"]
LARGE_CAP_THRESHOLD = 1.0
LARGE_IND_THRESHOLD = 3.0
UPPER_BOUND = 1e5


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


def s11_s21_ri_np(s_params: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            s_params[:, 0, 0].real,
            s_params[:, 0, 0].imag,
            s_params[:, 1, 0].real,
            s_params[:, 1, 0].imag,
        ]
    )


def db20(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def max_param_location(pred_row: pd.Series, train, wrapper, param_names: list[str]) -> tuple[str, int, float]:
    best_name = ""
    best_head = -1
    best_value = -np.inf
    for name in param_names:
        for head in range(1, wrapper.CONNECTION_COUNT + 1):
            value = float(pred_row[f"pred_conn{head}_{name}"])
            if value > best_value:
                best_name = name
                best_head = head
                best_value = value
    return best_name, best_head, best_value


def main() -> None:
    train = load_module(TRAIN_SCRIPT, "v11_positive_multihead_train_for_diag")
    source = train.load_module(train.SOURCE_SCRIPT, "v11_positive_multihead_diag_source")
    positive = train.load_module(train.POSITIVE_SCRIPT, "v11_positive_multihead_diag_positive")
    wrapper = train.load_module(train.WRAPPER_SCRIPT, "v11_positive_multihead_diag_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_positive_multihead_diag_base")

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

    opt_targets_all = pd.read_csv(opt_dir / train.OPT_TARGET_FILE, encoding="utf-8-sig")
    dut_all = positive.collect_lhs400_rdl_tsv_samples(base)
    target_ids = set(opt_targets_all["sample_id"].astype(str))
    dut_df = dut_all[dut_all["sample_id"].astype(str).isin(target_ids)].reset_index(drop=True)
    opt_targets = dut_df[["sample_id"]].merge(opt_targets_all, on="sample_id", how="left").set_index("sample_id")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = train.ADS_DEVICE_LENGTH_SCALE
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)

    pred = pd.read_csv(output_dir / "positive_symmetric_multihead_lc_fmin60_phase_predictions.csv", encoding="utf-8-sig").set_index("sample_id")
    metrics = pd.read_csv(output_dir / "positive_symmetric_multihead_lc_fmin60_phase_metrics.csv", encoding="utf-8-sig")
    columns = train.multihead_target_columns(wrapper)
    omega = 2.0 * np.pi * sim.freq_hz

    rows = []
    param_extreme_rows = []
    for i, row in dut_df.iterrows():
        sample_id = str(row["sample_id"])
        pred_row = pred.loc[sample_id]
        pred_params = pred_row[[f"pred_{col}" for col in columns]].to_numpy(dtype=np.float64)
        pred_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, pred_params))
        opt_params = opt_targets.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        opt_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, opt_params))

        pred_ri = s11_s21_ri_np(pred_s)
        opt_ri = s11_s21_ri_np(opt_s)
        pred_db = np.column_stack([db20(pred_s[:, 0, 0]), db20(pred_s[:, 1, 0])])
        opt_db = np.column_stack([db20(opt_s[:, 0, 0]), db20(opt_s[:, 1, 0])])
        pred_abs_d1 = np.abs(np.diff(pred_ri, axis=0))
        pred_db_d1_arr = np.abs(np.diff(pred_db, axis=0))
        pred_max_d1 = float(np.max(pred_abs_d1))
        pred_db_d1 = float(np.max(pred_db_d1_arr))
        pred_max_d1_idx = int(np.unravel_index(np.argmax(pred_abs_d1), pred_abs_d1.shape)[0])
        pred_db_d1_idx = int(np.unravel_index(np.argmax(pred_db_d1_arr), pred_db_d1_arr.shape)[0])

        max_c_name, max_c_head, max_c_value = max_param_location(pred_row, train, wrapper, CAP_PARAMS)
        max_l_name, max_l_head, max_l_value = max_param_location(pred_row, train, wrapper, IND_PARAMS)
        row_data = {
            "sample_id": sample_id,
            "split": row["split"],
            "pred_max_d1": pred_max_d1,
            "pred_db_d1": pred_db_d1,
            "pred_max_d1_freq_idx": pred_max_d1_idx,
            "pred_db_d1_freq_idx": pred_db_d1_idx,
            "pred_max_d1_freq_ghz": float(sim.freq_hz[min(pred_max_d1_idx + 1, len(sim.freq_hz) - 1)] / 1e9),
            "pred_db_d1_freq_ghz": float(sim.freq_hz[min(pred_db_d1_idx + 1, len(sim.freq_hz) - 1)] / 1e9),
            "opt_max_d1": float(np.max(np.abs(np.diff(opt_ri, axis=0)))),
            "opt_db_d1": float(np.max(np.abs(np.diff(opt_db, axis=0)))),
            "pred_is_resonant": bool(pred_db_d1 > DB_D1_THRESHOLD or pred_max_d1 > RI_D1_THRESHOLD),
            "max_cap_param": max_c_name,
            "max_cap_head": max_c_head,
            "max_cap_value": max_c_value,
            "max_ind_param": max_l_name,
            "max_ind_head": max_l_head,
            "max_ind_value": max_l_value,
            "large_cap_flag": bool(max_c_value > LARGE_CAP_THRESHOLD),
            "large_ind_flag": bool(max_l_value > LARGE_IND_THRESHOLD),
        }
        rows.append(row_data)

        for name in wrapper.V08_PARAM_NAMES:
            for head in range(1, wrapper.CONNECTION_COUNT + 1):
                value = float(pred_row[f"pred_conn{head}_{name}"])
                param_extreme_rows.append(
                    {
                        "sample_id": sample_id,
                        "split": row["split"],
                        "parameter": name,
                        "head": head,
                        "value": value,
                        "is_cap": name in CAP_PARAMS,
                        "is_ind": name in IND_PARAMS,
                        "near_upper_bound": bool(value >= 0.999 * UPPER_BOUND),
                    }
                )

    diag = pd.DataFrame(rows).merge(
        metrics[["sample_id", "nn_nmse_s11_s21_ri", "optimized_nmse_s11_s21_ri", "direct_nmse_s11_s21_ri"]],
        on="sample_id",
        how="left",
    )
    param_extremes = pd.DataFrame(param_extreme_rows)

    summary = (
        diag.groupby(["split", "pred_is_resonant"], as_index=False)
        .agg(
            count=("sample_id", "count"),
            pred_db_d1_mean=("pred_db_d1", "mean"),
            pred_max_d1_mean=("pred_max_d1", "mean"),
            nn_nmse_mean=("nn_nmse_s11_s21_ri", "mean"),
            max_cap_mean=("max_cap_value", "mean"),
            max_ind_mean=("max_ind_value", "mean"),
        )
    )
    param_summary = (
        param_extremes.groupby("parameter", as_index=False)
        .agg(
            count=("value", "count"),
            min=("value", "min"),
            p50=("value", "median"),
            p90=("value", lambda s: float(np.quantile(s, 0.90))),
            p95=("value", lambda s: float(np.quantile(s, 0.95))),
            p99=("value", lambda s: float(np.quantile(s, 0.99))),
            max=("value", "max"),
            near_upper_bound_count=("near_upper_bound", "sum"),
        )
    )
    resonant = diag[diag["pred_is_resonant"]].copy()
    resonant_param_summary = (
        param_extremes[param_extremes["sample_id"].isin(set(resonant["sample_id"]))]
        .groupby("parameter", as_index=False)
        .agg(
            resonant_count=("value", "count"),
            resonant_p95=("value", lambda s: float(np.quantile(s, 0.95))),
            resonant_p99=("value", lambda s: float(np.quantile(s, 0.99))),
            resonant_max=("value", "max"),
            resonant_near_upper_bound_count=("near_upper_bound", "sum"),
        )
    )
    worst = diag.sort_values(["pred_db_d1", "pred_max_d1"], ascending=False).head(30)
    large_lc = diag[diag["large_cap_flag"] | diag["large_ind_flag"]].sort_values(
        ["max_cap_value", "max_ind_value"],
        ascending=False,
    )
    top_extremes = param_extremes[param_extremes["parameter"].isin([*CAP_PARAMS, *IND_PARAMS])].sort_values(
        "value",
        ascending=False,
    ).head(100)

    diag.to_csv(output_dir / "symmetric_lc_fmin60_phase_resonance_lc_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "symmetric_lc_fmin60_phase_resonance_lc_summary.csv", index=False, encoding="utf-8-sig")
    param_summary.to_csv(output_dir / "symmetric_lc_fmin60_phase_parameter_range_diagnostic.csv", index=False, encoding="utf-8-sig")
    resonant_param_summary.to_csv(output_dir / "symmetric_lc_fmin60_phase_resonant_parameter_range_diagnostic.csv", index=False, encoding="utf-8-sig")
    worst.to_csv(output_dir / "symmetric_lc_fmin60_phase_worst_resonance_samples.csv", index=False, encoding="utf-8-sig")
    large_lc.to_csv(output_dir / "symmetric_lc_fmin60_phase_large_lc_samples.csv", index=False, encoding="utf-8-sig")
    top_extremes.to_csv(output_dir / "symmetric_lc_fmin60_phase_top_lc_parameter_outputs.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "active_sample_count": int(len(diag)),
        "resonant_count": int(diag["pred_is_resonant"].sum()),
        "large_cap_threshold": LARGE_CAP_THRESHOLD,
        "large_ind_threshold": LARGE_IND_THRESHOLD,
        "large_lc_sample_count": int(len(large_lc)),
        "summary": summary.to_dict(orient="records"),
        "parameter_summary": param_summary.to_dict(orient="records"),
        "resonant_parameter_summary": resonant_param_summary.to_dict(orient="records"),
        "worst_samples": worst.to_dict(orient="records"),
    }
    (output_dir / "symmetric_lc_fmin60_phase_resonance_lc_diagnosis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "symmetric_lc_fmin60_phase_resonance_lc_diagnosis_report.md").write_text(
        "\n".join(
            [
                "# Symmetric-LC fmin60 Phase Multi-Head Resonance and L/C Diagnosis",
                "",
                f"- Output: `{output_dir}`",
                f"- Active samples: `{len(diag)}`",
                f"- Resonant prediction samples: `{int(diag['pred_is_resonant'].sum())}`",
                f"- Large L/C flagged samples: `{len(large_lc)}`",
                f"- Resonance thresholds: `pred_db_d1 > {DB_D1_THRESHOLD}` or `pred_max_d1 > {RI_D1_THRESHOLD}`",
                f"- Large C threshold: `{LARGE_CAP_THRESHOLD}` on `Cn1/Cn2/Cn3` scale outputs",
                f"- Large L threshold: `{LARGE_IND_THRESHOLD}` on `Ln1` scale output",
                "",
                "## Resonance Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Range Summary",
                "",
                dataframe_to_markdown(param_summary),
                "",
                "## Resonant-Sample Parameter Range Summary",
                "",
                dataframe_to_markdown(resonant_param_summary),
                "",
                "## Worst Resonance Samples",
                "",
                dataframe_to_markdown(
                    worst[
                        [
                            "sample_id",
                            "split",
                            "pred_db_d1",
                            "pred_max_d1",
                            "nn_nmse_s11_s21_ri",
                            "max_cap_param",
                            "max_cap_head",
                            "max_cap_value",
                            "max_ind_head",
                            "max_ind_value",
                        ]
                    ]
                ),
                "",
                "## Outputs",
                "",
                f"- Per-sample diagnostics: `{output_dir / 'symmetric_lc_fmin60_phase_resonance_lc_diagnostics.csv'}`",
                f"- Large L/C samples: `{output_dir / 'symmetric_lc_fmin60_phase_large_lc_samples.csv'}`",
                f"- Top L/C outputs: `{output_dir / 'symmetric_lc_fmin60_phase_top_lc_parameter_outputs.csv'}`",
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)
    print(dataframe_to_markdown(param_summary), flush=True)
    print(dataframe_to_markdown(worst[["sample_id", "split", "pred_db_d1", "pred_max_d1", "nn_nmse_s11_s21_ri", "max_cap_param", "max_cap_head", "max_cap_value", "max_ind_head", "max_ind_value"]]), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
