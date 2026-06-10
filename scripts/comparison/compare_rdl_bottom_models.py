import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio
import skrf as rf

matplotlib.use("Agg")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rdl_tsv_transition.plotting import (
    db20,
    save_model_case_plot,
    save_model_summary_plots,
)


TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
FEATURE_ORDER = ["ldown", "wdown", "tdown", "htsv", "p1"]
CSV_FEATURE_ORDER = ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl"]
Z_REF = 50.0


def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).stem)
    return [int(part) if part.isdigit() else part for part in parts]


def parse_touchstone_variables(path):
    params = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                break
            if not line.startswith("!") or "=" not in line:
                continue
            key, value = line[1:].split("=", 1)
            match = re.search(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", value)
            if match:
                params[key.strip()] = float(match.group(1))

    missing = [name for name in FEATURE_ORDER if name not in params]
    if missing:
        raise ValueError(f"{path} missing variables in header: {missing}")
    return params


def csv_case_keys(case_csv):
    if not case_csv:
        return None
    df = pd.read_csv(case_csv)
    missing = [name for name in CSV_FEATURE_ORDER if name not in df.columns]
    if missing:
        raise ValueError(f"{case_csv} missing columns: {missing}")
    keys = set()
    for _, row in df.iterrows():
        keys.add(tuple(round(float(row[name]), 6) for name in CSV_FEATURE_ORDER))
    return keys


def touchstone_key(variables):
    return tuple(round(float(variables[name]), 6) for name in FEATURE_ORDER)


def load_matlab_model(config, base_dir):
    model_dir = (base_dir / config["path"]).resolve()
    prefix = config.get("prefix", "RDL_Bottom_")
    missing = [name for name in TARGET_PARAMS if not (model_dir / f"{prefix}{name}.mat").exists()]
    if missing:
        if config.get("required", False):
            raise FileNotFoundError(f"{config['label']} missing {prefix}*.mat files: {missing}")
        print(f"[skip] {config['label']}: missing {prefix}*.mat files: {missing}")
        return None

    dims = {}
    for name in TARGET_PARAMS:
        data = sio.loadmat(model_dir / f"{prefix}{name}.mat")
        dims[name] = int(np.asarray(data["psmin"]).size)

    unique_dims = sorted(set(dims.values()))
    if len(unique_dims) != 1 or unique_dims[0] not in (3, 5):
        raise ValueError(f"{model_dir} has unsupported input dimensions: {dims}")

    input_dim = unique_dims[0]
    return {
        "type": "matlab",
        "label": config["label"],
        "dir": model_dir,
        "prefix": prefix,
        "feature_names": FEATURE_ORDER[:input_dim],
    }


def load_torch_model(config, base_dir):
    model_dir = (base_dir / config["path"]).resolve()
    filename = config.get("filename", "rdl_bottom_param_net.pt")
    pt_path = model_dir / filename
    if not pt_path.exists():
        if config.get("required", False):
            raise FileNotFoundError(f"{config['label']} missing model file: {pt_path}")
        print(f"[skip] {config['label']}: missing model file: {pt_path}")
        return None

    import torch
    import torch.nn as nn

    class ParamNet(nn.Module):
        def __init__(self, in_dim=5, out_dim=9, hidden=96):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = ParamNet(
        in_dim=config.get("in_dim", 5),
        out_dim=config.get("out_dim", 9),
        hidden=config.get("hidden", 96),
    ).double()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "type": "torch",
        "label": config["label"],
        "dir": model_dir,
        "model": model,
        "torch": torch,
        "stats": {k: np.asarray(v, dtype=float) for k, v in checkpoint["stats"].items()},
        "feature_names": config.get("feature_names", FEATURE_ORDER),
    }


def load_model(config, base_dir):
    if config["type"] == "matlab":
        return load_matlab_model(config, base_dir)
    if config["type"] == "torch":
        return load_torch_model(config, base_dir)
    raise ValueError(f"Unsupported model type: {config['type']}")


def predict_matlab_parameters(variables, meta):
    features = np.array([variables[name] for name in meta["feature_names"]], dtype=float).reshape(1, -1)
    circuit_params = {}
    for name in TARGET_PARAMS:
        data = sio.loadmat(meta["dir"] / f"{meta['prefix']}{name}.mat")
        xmin = np.asarray(data["psmin"], dtype=float)
        xmax = np.asarray(data["psmax"], dtype=float)
        ymin = float(np.asarray(data["outputmin"]).squeeze())
        ymax = float(np.asarray(data["outputmax"]).squeeze())
        w1 = np.asarray(data["w1"], dtype=float)
        b1 = np.asarray(data["theta1"], dtype=float)
        w2 = np.asarray(data["w2"], dtype=float)
        b2 = np.asarray(data["theta2"], dtype=float)
        w3 = np.asarray(data["w3"], dtype=float)
        b3 = np.asarray(data["theta3"], dtype=float)

        x_norm = 2.0 * (features - xmin) / (xmax - xmin + 1e-12) - 1.0
        a1 = np.tanh(x_norm @ w1 + b1)
        a2 = np.tanh(a1 @ w2 + b2)
        y_norm = a2 @ w3 + b3
        y_real = ymin + (y_norm + 1.0) * (ymax - ymin) / 2.0
        circuit_params[name] = float(y_real.squeeze())
    return circuit_params


def predict_torch_parameters(variables, meta):
    features = np.array([variables[name] for name in meta["feature_names"]], dtype=float).reshape(1, -1)
    stats = meta["stats"]
    x_norm = (features - stats["x_mean"]) / (stats["x_std"] + 1e-12)
    with meta["torch"].no_grad():
        y_norm = meta["model"](meta["torch"].tensor(x_norm, dtype=meta["torch"].float64)).numpy()
    y_log = y_norm * stats["y_std"] + stats["y_mean"]
    y = np.exp(y_log).reshape(-1)
    return {name: float(value) for name, value in zip(TARGET_PARAMS, y)}


def predict_parameters(variables, meta):
    if meta["type"] == "matlab":
        return predict_matlab_parameters(variables, meta)
    if meta["type"] == "torch":
        return predict_torch_parameters(variables, meta)
    raise ValueError(f"Unsupported model type: {meta['type']}")


def calculate_s_parameters(circuit_params, length_um, freqs_hz):
    R1, R2, R3 = circuit_params["R1"], circuit_params["R2"], circuit_params["R3"]
    L1, L2, L3 = circuit_params["L1"] * 1e-9, circuit_params["L2"] * 1e-9, circuit_params["L3"] * 1e-9
    Cox, Csi, Rsi = circuit_params["Cox"] * 1e-12, circuit_params["Csi"] * 1e-12, circuit_params["Rsi"]
    length_m = length_um * 1e-6
    omega = 2.0 * np.pi * freqs_hz

    r_rlgc = (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2) / (
        (R1 + R2) ** 2 + omega**2 * L2**2
    ) + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    l_rlgc = (R1**2 * L2) / ((R1 + R2) ** 2 + omega**2 * L2**2) + (
        L3 * R3**2
    ) / (R3**2 + omega**2 * L3**2) + L1
    g_rlgc = (omega**2 * Rsi * Cox**2) / (1.0 + omega**2 * Rsi**2 * (Cox + Csi) ** 2)
    c_rlgc = (Cox + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)) / (
        1.0 + omega**2 * Rsi**2 * (Cox + Csi) ** 2
    )

    z0 = np.sqrt((r_rlgc + 1j * omega * l_rlgc) / (g_rlgc + 1j * omega * c_rlgc))
    gamma = np.sqrt((r_rlgc + 1j * omega * l_rlgc) * (g_rlgc + 1j * omega * c_rlgc))
    a = np.cosh(gamma * length_m)
    b = z0 * np.sinh(gamma * length_m)
    c = np.sinh(gamma * length_m) / z0
    d = np.cosh(gamma * length_m)

    denom = a + b / Z_REF + c * Z_REF + d
    s = np.empty((len(freqs_hz), 2, 2), dtype=complex)
    s[:, 0, 0] = (a + b / Z_REF - c * Z_REF - d) / denom
    s[:, 0, 1] = 2.0 * (a * d - b * c) / denom
    s[:, 1, 0] = 2.0 / denom
    s[:, 1, 1] = (-a + b / Z_REF - c * Z_REF + d) / denom
    return s


def safe_rel_error(pred, ref):
    return np.abs(pred - ref) / np.maximum(np.abs(ref), 1e-12)


def calc_metrics(pred_s, ref_s, label_prefix):
    diff = pred_s - ref_s
    metrics = {
        f"{label_prefix}_complex_mse": float(np.mean(np.abs(diff) ** 2)),
        f"{label_prefix}_complex_rmse": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
        f"{label_prefix}_complex_mae": float(np.mean(np.abs(diff))),
        f"{label_prefix}_rel_mae": float(np.mean(safe_rel_error(pred_s, ref_s))),
    }
    for m, n, name in [(0, 0, "s11"), (1, 0, "s21"), (0, 1, "s12"), (1, 1, "s22")]:
        pred = pred_s[:, m, n]
        ref = ref_s[:, m, n]
        metrics[f"{label_prefix}_{name}_db_mae"] = float(np.mean(np.abs(db20(pred) - db20(ref))))
        metrics[f"{label_prefix}_{name}_db_max"] = float(np.max(np.abs(db20(pred) - db20(ref))))
        metrics[f"{label_prefix}_{name}_phase_mae_deg"] = float(
            np.mean(np.abs(np.unwrap(np.angle(pred)) - np.unwrap(np.angle(ref)))) * 180.0 / np.pi
        )
    return metrics


def build_model_specs(model_configs, base_dir):
    models = []
    for config in model_configs:
        meta = load_model(config, base_dir)
        if meta is not None:
            models.append((config["label"], meta))
    if not models:
        raise RuntimeError("No comparable models were loaded.")
    return models


def load_s2p_files(hfss_dir, case_keys, limit):
    s2p_files = sorted(hfss_dir.glob("*.s2p"), key=natural_key)
    if case_keys is not None:
        filtered = []
        for s2p_file in s2p_files:
            variables = parse_touchstone_variables(s2p_file)
            if touchstone_key(variables) in case_keys:
                filtered.append(s2p_file)
        s2p_files = filtered
    if limit:
        s2p_files = s2p_files[:limit]
    if not s2p_files:
        raise FileNotFoundError(f"No .s2p files found under {hfss_dir}")
    return s2p_files


def write_aggregate_outputs(summary_df, out_dir, model_names):
    aggregate_rows = []
    metric_cols = [
        col
        for col in summary_df.columns
        if col.endswith(("_mse", "_rmse", "_mae", "_db_mae", "_db_max", "_phase_mae_deg"))
    ]
    for col in metric_cols:
        aggregate_rows.append(
            {
                "metric": col,
                "mean": summary_df[col].mean(),
                "median": summary_df[col].median(),
                "min": summary_df[col].min(),
                "max": summary_df[col].max(),
                "worst_file": summary_df.loc[summary_df[col].idxmax(), "file"] if len(summary_df) else None,
            }
        )

    aggregate_csv = out_dir / "rdl_bottom_model_compare_aggregate.csv"
    pd.DataFrame(aggregate_rows).to_csv(aggregate_csv, index=False)

    compact_rows = []
    for model_name in model_names:
        compact_rows.append(
            {
                "model": model_name,
                "mean_complex_mse": summary_df[f"{model_name}_vs_hfss_complex_mse"].mean(),
                "median_complex_mse": summary_df[f"{model_name}_vs_hfss_complex_mse"].median(),
                "max_complex_mse": summary_df[f"{model_name}_vs_hfss_complex_mse"].max(),
                "mean_s11_db_mae": summary_df[f"{model_name}_vs_hfss_s11_db_mae"].mean(),
                "mean_s21_db_mae": summary_df[f"{model_name}_vs_hfss_s21_db_mae"].mean(),
                "mean_s11_phase_mae_deg": summary_df[f"{model_name}_vs_hfss_s11_phase_mae_deg"].mean(),
                "mean_s21_phase_mae_deg": summary_df[f"{model_name}_vs_hfss_s21_phase_mae_deg"].mean(),
            }
        )
    compact_df = pd.DataFrame(compact_rows).sort_values("mean_complex_mse")
    compact_csv = out_dir / "rdl_bottom_model_compare_compact.csv"
    compact_df.to_csv(compact_csv, index=False)
    return aggregate_csv, compact_csv, compact_df


def compare_models(args, model_configs, reference_model=None):
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[2]
    hfss_dir = (base_dir / args.hfss_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    case_keys = csv_case_keys(base_dir / args.case_csv) if args.case_csv else None
    model_specs = build_model_specs(model_configs, base_dir)
    model_names = [name for name, _ in model_specs]
    s2p_files = load_s2p_files(hfss_dir, case_keys, args.limit)

    rows = []
    worst = {name: (-np.inf, None, None, None) for name in model_names}
    for idx, s2p_file in enumerate(s2p_files, start=1):
        try:
            variables = parse_touchstone_variables(s2p_file)
            nw_hfss = rf.Network(str(s2p_file))
            length_um = variables[args.length_param]
            row = {
                "file": s2p_file.name,
                "length_param": args.length_param,
                "length_um": length_um,
                **{name: variables[name] for name in FEATURE_ORDER},
            }

            pred_by_model = {}
            for model_name, meta in model_specs:
                params = predict_parameters(variables, meta)
                pred_s = calculate_s_parameters(params, length_um, nw_hfss.f)
                pred_by_model[model_name] = pred_s
                row[f"{model_name}_features"] = ",".join(meta["feature_names"])
                row.update({f"{model_name}_{key}": value for key, value in params.items()})
                row.update(calc_metrics(pred_s, nw_hfss.s, f"{model_name}_vs_hfss"))

                mse = row[f"{model_name}_vs_hfss_complex_mse"]
                if mse > worst[model_name][0]:
                    worst[model_name] = (mse, s2p_file.name, nw_hfss, pred_by_model.copy())

            if reference_model in pred_by_model:
                for model_name, pred_s in pred_by_model.items():
                    if model_name != reference_model:
                        row.update(calc_metrics(pred_s, pred_by_model[reference_model], f"{model_name}_vs_{reference_model}"))

            rows.append(row)
            if not args.no_plots and (args.plot_all or (args.plot_first and idx <= args.plot_first)):
                save_model_case_plot(
                    plots_dir / f"{s2p_file.stem}_comparison.png",
                    nw_hfss,
                    pred_by_model,
                    f"{s2p_file.name} RDL_Bottom model comparison",
                )
            if idx % args.progress_every == 0:
                print(f"Processed {idx}/{len(s2p_files)}: {s2p_file.name}")
        except Exception as exc:
            rows.append({"file": s2p_file.name, "error": str(exc)})
            print(f"[skip] {s2p_file.name}: {exc}")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "rdl_bottom_model_compare_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    valid = summary_df[summary_df.get("error").isna()] if "error" in summary_df else summary_df
    aggregate_csv, compact_csv, compact_df = write_aggregate_outputs(valid, out_dir, model_names)

    if not args.no_plots and len(valid):
        save_model_summary_plots(out_dir, valid.reset_index(drop=True), model_names)
        for model_name, (mse, filename, nw_hfss, pred_by_model) in worst.items():
            if filename is None:
                continue
            save_model_case_plot(
                plots_dir / f"worst_{model_name}_{Path(filename).stem}.png",
                nw_hfss,
                pred_by_model,
                f"Worst {model_name}: {filename}, MSE={mse:.3e}",
            )

    with open(out_dir / "compare_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_cases": int(len(valid)),
                "n_total": int(len(summary_df)),
                "models": model_names,
                "summary_csv": str(summary_csv),
                "aggregate_csv": str(aggregate_csv),
                "compact_csv": str(compact_csv),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nCompare complete")
    print(f"  HFSS dir: {hfss_dir}")
    print(f"  valid cases: {len(valid)} / {len(summary_df)}")
    print(f"  models: {', '.join(model_names)}")
    print(f"  summary CSV: {summary_csv}")
    print(f"  compact CSV: {compact_csv}")
    if not args.no_plots:
        print(f"  plot dir: {plots_dir}")
    if len(compact_df):
        print(compact_df.to_string(index=False))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare RDL_Bottom MATLAB .mat models and PyTorch .pt models against HFSS s2p files."
    )
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--hfss-dir", default="data/sparameters/RDL_Bottom_Snp")
    parser.add_argument("--case-csv", default="", help="Optional CSV used to filter geometry cases.")
    parser.add_argument("--out-dir", default="outputs/comparison/RDL_Bottom_model_compare")
    parser.add_argument("--length-param", choices=["htsv", "ldown"], default="ldown")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-all", action="store_true")
    parser.add_argument("--plot-first", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main():
    args = build_arg_parser().parse_args()

    # Define all models to compare here.
    model_configs = [
        {"label": "mat1", "type": "matlab", "path": "data/matlab_models/RDL_TSV_mat1"},
        {"label": "mat2", "type": "matlab", "path": "data/matlab_models/RDL_TSV_mat2"},
        {"label": "mat3", "type": "matlab", "path": "data/matlab_models/RDL_TSV_mat3"},
        {"label": "mat4", "type": "matlab", "path": "data/matlab_models/RDL_TSV_mat4"},
        {
            "label": "new_s_finetuned",
            "type": "torch",
            "path": "outputs/training/RDL_Bottom_TD4_trend_sparam_training",
            "filename": "rdl_bottom_param_net.pt",
        },
    ]

    compare_models(args, model_configs, reference_model="mat2")


if __name__ == "__main__":
    main()
