# -*- coding: utf-8 -*-
"""Compare resonant NN outputs with optimized circuit-parameter maxima.

Run this file directly in VS Code. No command-line arguments are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERSION_ROOT = PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
RESULT_ROOT = VERSION_ROOT / "results"
OPT_DIR = RESULT_ROOT / "v11_positive_sharedopt_lhs400_connection2_goodstart_all_adslen09"
OPT_FILE = OPT_DIR / "v08_positive_goodstart_targets.csv"
OUT_DIR = RESULT_ROOT / "v11_positive_resonant_param_exceed_analysis"

PARAMS = [
    "Cn1_scale",
    "Rn1_scale",
    "Cn2_scale",
    "Rn2_scale",
    "Cn3_scale",
    "Rn3_scale",
    "Ln1_scale",
]
CAP_PARAMS = ["Cn1_scale", "Cn2_scale", "Cn3_scale"]
RES_PARAMS = ["Rn1_scale", "Rn2_scale", "Rn3_scale"]
IND_PARAMS = ["Ln1_scale"]
GROUPS = {"C": CAP_PARAMS, "R": RES_PARAMS, "L": IND_PARAMS}
SCALE_TO_PHYSICAL = {
    "Cn1_scale": 1e-14,
    "Cn2_scale": 1e-14,
    "Cn3_scale": 1e-14,
    "Ln1_scale": 1e-11,
    "Rn1_scale": 1.0,
    "Rn2_scale": 1.0,
    "Rn3_scale": 1.0,
}

RUNS = [
    {
        "run_label": "fmin60",
        "result_dir": "v11_positive_symmetric_multihead_lc_fmin60_log_adslen09",
        "prediction_file": "positive_symmetric_multihead_lc_fmin60_predictions.csv",
        "diagnostic_file": "symmetric_lc_fmin60_resonance_lc_diagnostics.csv",
    },
    {
        "run_label": "fmin60_phase",
        "result_dir": "v11_positive_symmetric_multihead_lc_fmin60_phase_log_adslen09",
        "prediction_file": "positive_symmetric_multihead_lc_fmin60_phase_predictions.csv",
        "diagnostic_file": "symmetric_lc_fmin60_phase_resonance_lc_diagnostics.csv",
    },
    {
        "run_label": "fmin100_phase",
        "result_dir": "v11_positive_symmetric_multihead_lc_fmin100_phase_log_adslen09",
        "prediction_file": "positive_symmetric_multihead_lc_fmin100_phase_predictions.csv",
        "diagnostic_file": "symmetric_lc_fmin100_phase_resonance_lc_diagnostics.csv",
    },
    {
        "run_label": "fmin60_phase_spike",
        "result_dir": "v11_positive_symmetric_multihead_lc_fmin60_phase_spike_log_adslen09",
        "prediction_file": "positive_symmetric_multihead_lc_fmin60_phase_spike_predictions.csv",
        "diagnostic_file": "symmetric_lc_fmin60_phase_spike_resonance_lc_diagnostics.csv",
    },
]


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
            values.append(f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def prediction_columns(param: str) -> list[str]:
    return [f"pred_conn{idx}_{param}" for idx in range(1, 13)]


def optimized_parameter_max(opt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for param in PARAMS:
        idx = opt[param].astype(float).idxmax()
        scale_max = float(opt.loc[idx, param])
        rows.append(
            {
                "parameter": param,
                "group": "C" if param in CAP_PARAMS else "R" if param in RES_PARAMS else "L",
                "scale_max": scale_max,
                "physical_max": scale_max * SCALE_TO_PHYSICAL[param],
                "sample_id_at_max": str(opt.loc[idx, "sample_id"]),
            }
        )
    return pd.DataFrame(rows)


def optimized_group_max(param_max: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, params in GROUPS.items():
        sub = param_max[param_max["parameter"].isin(params)].copy()
        idx = sub["physical_max"].idxmax()
        rows.append(
            {
                "group": group,
                "physical_max": float(sub.loc[idx, "physical_max"]),
                "scale_max_at_group_max": float(sub.loc[idx, "scale_max"]),
                "parameter_at_group_max": str(sub.loc[idx, "parameter"]),
                "sample_id_at_group_max": str(sub.loc[idx, "sample_id_at_max"]),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(opt: pd.DataFrame, param_max: pd.DataFrame, group_max: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    param_max_map = param_max.set_index("parameter")["scale_max"].to_dict()
    group_physical_max_map = group_max.set_index("group")["physical_max"].to_dict()
    opt_by_sample = opt.set_index("sample_id")
    sample_rows = []
    param_rows = []

    for cfg in RUNS:
        run_dir = RESULT_ROOT / cfg["result_dir"]
        pred_path = run_dir / cfg["prediction_file"]
        diag_path = run_dir / cfg["diagnostic_file"]
        if not pred_path.exists() or not diag_path.exists():
            continue
        pred = pd.read_csv(pred_path, encoding="utf-8-sig").set_index("sample_id")
        diag = pd.read_csv(diag_path, encoding="utf-8-sig")
        resonant = diag[diag["pred_is_resonant"].astype(bool)].copy()

        for _, diag_row in resonant.iterrows():
            sid = str(diag_row["sample_id"])
            if sid not in pred.index:
                continue
            pred_row = pred.loc[sid]
            opt_row = opt_by_sample.loc[sid] if sid in opt_by_sample.index else None

            sample_record = {
                "run_label": cfg["run_label"],
                "sample_id": sid,
                "pred_db_d1": float(diag_row.get("pred_db_d1", np.nan)),
                "pred_max_d1": float(diag_row.get("pred_max_d1", np.nan)),
                "nn_nmse_s11_s21_ri": float(diag_row.get("nn_nmse_s11_s21_ri", np.nan)),
            }
            sample_exceeded_params: list[str] = []

            for group, params in GROUPS.items():
                group_values = []
                for param in params:
                    vals = pred_row[prediction_columns(param)].to_numpy(dtype=np.float64)
                    group_values.extend((vals * SCALE_TO_PHYSICAL[param]).tolist())
                group_max_value = float(np.max(group_values))
                opt_group_max_value = float(group_physical_max_map[group])
                sample_record[f"nn_{group}_physical_max"] = group_max_value
                sample_record[f"optimized_{group}_physical_global_max"] = opt_group_max_value
                sample_record[f"{group}_exceeds_optimized_global_max"] = bool(group_max_value > opt_group_max_value)
                sample_record[f"{group}_ratio_to_optimized_global_max"] = group_max_value / opt_group_max_value if opt_group_max_value else np.nan

            for param in PARAMS:
                vals = pred_row[prediction_columns(param)].to_numpy(dtype=np.float64)
                nn_scale_max = float(np.max(vals))
                opt_global_max = float(param_max_map[param])
                same_sample_opt = float(opt_row[param]) if opt_row is not None else np.nan
                exceeds = bool(nn_scale_max > opt_global_max)
                if exceeds:
                    sample_exceeded_params.append(param)
                param_rows.append(
                    {
                        "run_label": cfg["run_label"],
                        "sample_id": sid,
                        "parameter": param,
                        "group": "C" if param in CAP_PARAMS else "R" if param in RES_PARAMS else "L",
                        "nn_scale_max": nn_scale_max,
                        "optimized_global_scale_max": opt_global_max,
                        "exceeds_optimized_global_max": exceeds,
                        "ratio_to_optimized_global_max": nn_scale_max / opt_global_max if opt_global_max else np.nan,
                        "same_sample_optimized_scale": same_sample_opt,
                        "ratio_to_same_sample_optimized": nn_scale_max / same_sample_opt if same_sample_opt else np.nan,
                        "nn_physical_max": nn_scale_max * SCALE_TO_PHYSICAL[param],
                        "optimized_global_physical_max": opt_global_max * SCALE_TO_PHYSICAL[param],
                    }
                )

            sample_record["exceeded_parameters"] = ",".join(sample_exceeded_params) if sample_exceeded_params else "none"
            sample_record["any_parameter_exceeds_optimized_global_max"] = bool(sample_exceeded_params)
            sample_rows.append(sample_record)

    return pd.DataFrame(sample_rows), pd.DataFrame(param_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opt = pd.read_csv(OPT_FILE, encoding="utf-8-sig")
    param_max = optimized_parameter_max(opt)
    group_max = optimized_group_max(param_max)
    sample_compare, param_compare = run_analysis(opt, param_max, group_max)

    param_max.to_csv(OUT_DIR / "optimized_parameter_max_stats.csv", index=False, encoding="utf-8-sig")
    group_max.to_csv(OUT_DIR / "optimized_group_max_stats.csv", index=False, encoding="utf-8-sig")
    sample_compare.to_csv(OUT_DIR / "resonant_sample_nn_vs_optimized_max.csv", index=False, encoding="utf-8-sig")
    param_compare.to_csv(OUT_DIR / "resonant_parameter_nn_vs_optimized_max.csv", index=False, encoding="utf-8-sig")

    run_summary = (
        sample_compare.groupby("run_label")
        .agg(
            resonant_sample_count=("sample_id", "count"),
            samples_with_any_param_exceed=("any_parameter_exceeds_optimized_global_max", "sum"),
            samples_with_C_exceed=("C_exceeds_optimized_global_max", "sum"),
            samples_with_R_exceed=("R_exceeds_optimized_global_max", "sum"),
            samples_with_L_exceed=("L_exceeds_optimized_global_max", "sum"),
            mean_nn_nmse=("nn_nmse_s11_s21_ri", "mean"),
        )
        .reset_index()
        if len(sample_compare)
        else pd.DataFrame()
    )
    run_summary.to_csv(OUT_DIR / "resonant_exceed_run_summary.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "# V11 Resonant NN Output vs Optimized Parameter Max",
        "",
        f"- Optimized target file: `{OPT_FILE}`",
        f"- Resonant runs checked: `{len(RUNS)}`",
        "",
        "## Optimized Parameter Maxima",
        "",
        dataframe_to_markdown(param_max),
        "",
        "## Optimized Group Maxima",
        "",
        dataframe_to_markdown(group_max),
        "",
        "## Resonant Sample Summary",
        "",
        dataframe_to_markdown(run_summary) if len(run_summary) else "No resonant samples found.",
        "",
        "## Interpretation",
        "",
        "- `exceeds_optimized_global_max` checks whether any NN output for a resonant sample is outside the global optimized-target range.",
        "- `ratio_to_same_sample_optimized` is often more informative for branch-placement errors: a value can be far above the same sample's optimized value while still below the global maximum.",
    ]
    (OUT_DIR / "resonant_param_exceed_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(dataframe_to_markdown(group_max), flush=True)
    print(dataframe_to_markdown(run_summary), flush=True)
    print(f"Done: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
