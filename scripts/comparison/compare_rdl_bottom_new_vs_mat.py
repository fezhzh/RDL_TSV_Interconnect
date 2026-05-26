import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import skrf as rf
import torch
import torch.nn as nn


TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
FEATURE_ORDER = ["ldown", "wdown", "tdown", "htsv", "p1"]
CSV_FEATURE_ORDER = ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl"]
Z_REF = 50.0


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
        raise ValueError(f"{path} 缺少变量注释: {missing}")
    return params


def csv_case_keys(case_csv):
    if not case_csv:
        return None
    df = pd.read_csv(case_csv)
    keys = set()
    for _, row in df.iterrows():
        keys.add(tuple(round(float(row[name]), 6) for name in CSV_FEATURE_ORDER))
    return keys


def touchstone_key(variables):
    return tuple(round(float(variables[name]), 6) for name in FEATURE_ORDER)


def load_mat_model_meta(mat_dir, label, prefix="RDL_Bottom_"):
    mat_dir = Path(mat_dir)
    missing = [name for name in TARGET_PARAMS if not (mat_dir / f"{prefix}{name}.mat").exists()]
    if missing:
        print(f"[skip] {label}: 缺少 {prefix}*.mat: {missing}")
        return None
    dims = {}
    for name in TARGET_PARAMS:
        data = sio.loadmat(mat_dir / f"{prefix}{name}.mat")
        dims[name] = int(np.asarray(data["psmin"]).size)
    unique_dims = sorted(set(dims.values()))
    if len(unique_dims) != 1 or unique_dims[0] not in (3, 5):
        raise ValueError(f"{mat_dir} 输入维度异常: {dims}")
    return {
        "type": "matlab",
        "label": label,
        "dir": mat_dir,
        "prefix": prefix,
        "input_dim": unique_dims[0],
        "feature_names": FEATURE_ORDER[: unique_dims[0]],
    }


def load_new_model_meta(model_dir, label="new_s_finetuned"):
    model_dir = Path(model_dir)
    pt_path = model_dir / "rdl_bottom_param_net.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"未找到新模型: {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = ParamNet().double()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = checkpoint["stats"]
    return {
        "type": "torch",
        "label": label,
        "dir": model_dir,
        "model": model,
        "stats": {k: np.asarray(v, dtype=float) for k, v in stats.items()},
        "feature_names": FEATURE_ORDER,
    }


def predict_matlab_parameters(features, model_meta):
    x = np.asarray(features, dtype=float).reshape(1, -1)
    circuit_params = {}
    for name in TARGET_PARAMS:
        data = sio.loadmat(model_meta["dir"] / f"{model_meta['prefix']}{name}.mat")
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
        x_norm = 2.0 * (x - xmin) / (xmax - xmin + 1e-12) - 1.0
        a1 = np.tanh(x_norm @ w1 + b1)
        a2 = np.tanh(a1 @ w2 + b2)
        y_norm = a2 @ w3 + b3
        y_real = ymin + (y_norm + 1.0) * (ymax - ymin) / 2.0
        circuit_params[name] = float(y_real.squeeze())
    return circuit_params


def predict_torch_parameters(features, model_meta):
    stats = model_meta["stats"]
    x = np.asarray(features, dtype=float).reshape(1, -1)
    x_norm = (x - stats["x_mean"]) / (stats["x_std"] + 1e-12)
    with torch.no_grad():
        y_norm = model_meta["model"](torch.tensor(x_norm, dtype=torch.float64)).numpy()
    y_log = y_norm * stats["y_std"] + stats["y_mean"]
    y = np.exp(y_log).reshape(-1)
    return {name: float(value) for name, value in zip(TARGET_PARAMS, y)}


def predict_parameters(variables, model_meta):
    features = np.array([variables[name] for name in model_meta["feature_names"]], dtype=float)
    if model_meta["type"] == "matlab":
        return predict_matlab_parameters(features, model_meta)
    return predict_torch_parameters(features, model_meta)


def calculate_s_parameters(circuit_params, length_um, freqs_hz):
    R1, R2, R3 = circuit_params["R1"], circuit_params["R2"], circuit_params["R3"]
    L1, L2, L3 = circuit_params["L1"] * 1e-9, circuit_params["L2"] * 1e-9, circuit_params["L3"] * 1e-9
    Cox, Csi, Rsi = circuit_params["Cox"] * 1e-12, circuit_params["Csi"] * 1e-12, circuit_params["Rsi"]
    length_m = length_um * 1e-6
    omega = 2.0 * np.pi * freqs_hz
    R_rlgc = (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2) / (
        (R1 + R2) ** 2 + omega**2 * L2**2
    ) + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    L_rlgc = (R1**2 * L2) / ((R1 + R2) ** 2 + omega**2 * L2**2) + (
        L3 * R3**2
    ) / (R3**2 + omega**2 * L3**2) + L1
    G_rlgc = (omega**2 * Rsi * Cox**2) / (1.0 + omega**2 * Rsi**2 * (Cox + Csi) ** 2)
    C_rlgc = (Cox + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)) / (
        1.0 + omega**2 * Rsi**2 * (Cox + Csi) ** 2
    )
    z0 = np.sqrt((R_rlgc + 1j * omega * L_rlgc) / (G_rlgc + 1j * omega * C_rlgc))
    gamma = np.sqrt((R_rlgc + 1j * omega * L_rlgc) * (G_rlgc + 1j * omega * C_rlgc))
    A = np.cosh(gamma * length_m)
    B = z0 * np.sinh(gamma * length_m)
    C = np.sinh(gamma * length_m) / z0
    D = np.cosh(gamma * length_m)
    denom = A + B / Z_REF + C * Z_REF + D
    s = np.empty((len(freqs_hz), 2, 2), dtype=complex)
    s[:, 0, 0] = (A + B / Z_REF - C * Z_REF - D) / denom
    s[:, 0, 1] = 2.0 * (A * D - B * C) / denom
    s[:, 1, 0] = 2.0 / denom
    s[:, 1, 1] = (-A + B / Z_REF - C * Z_REF + D) / denom
    return s


def db20(value):
    return 20.0 * np.log10(np.maximum(np.abs(value), 1e-300))


def calc_metrics(pred_s, ref_s, label_prefix):
    diff = pred_s - ref_s
    metrics = {
        f"{label_prefix}_complex_mse": float(np.mean(np.abs(diff) ** 2)),
        f"{label_prefix}_complex_rmse": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
        f"{label_prefix}_complex_mae": float(np.mean(np.abs(diff))),
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


def configure_matplotlib():
    plt.rcParams.update({"figure.facecolor": "#f6f8fb", "axes.facecolor": "white", "grid.color": "#e2e8f0"})


def save_case_plot(out_path, nw_hfss, pred_by_model, title):
    configure_matplotlib()
    freq_ghz = nw_hfss.f / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=16, fontweight="semibold")
    ports = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]
    for ax, (m, n, name) in zip(axes.ravel(), ports):
        ax.plot(freq_ghz, db20(nw_hfss.s[:, m, n]), label="HFSS", linewidth=1.8)
        for model_name, pred_s in pred_by_model.items():
            ax.plot(freq_ghz, db20(pred_s[:, m, n]), "--", label=model_name, linewidth=1.5)
        ax.set_title(f"{name} magnitude", loc="left")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def save_summary_plots(out_dir, summary_df, model_names):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    x = np.arange(len(summary_df))
    for model in model_names:
        axes[0].plot(x, summary_df[f"{model}_vs_hfss_complex_mse"].to_numpy(float), label=model)
        axes[1].plot(x, summary_df[f"{model}_vs_hfss_s21_db_mae"].to_numpy(float), label=model)
    axes[0].set_yscale("log")
    axes[0].set_title("Complex S MSE vs HFSS")
    axes[1].set_title("S21 magnitude MAE vs HFSS")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_error_trends.png")
    plt.close(fig)


def build_models(args, base_dir):
    models = []
    for spec in args.mat_model:
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).name, spec)
        meta = load_mat_model_meta(base_dir / path, label)
        if meta is not None:
            models.append((label, meta))
    if args.new_model_dir:
        models.append((args.new_label, load_new_model_meta(base_dir / args.new_model_dir, args.new_label)))
    if not models:
        raise RuntimeError("没有可比较的模型。")
    return models


def compare_models(args):
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[2]
    hfss_dir = (base_dir / args.hfss_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    case_keys = csv_case_keys(base_dir / args.case_csv) if args.case_csv else None
    model_specs = build_models(args, base_dir)
    model_names = [name for name, _ in model_specs]

    s2p_files = sorted(hfss_dir.glob("*.s2p"), key=natural_key)
    if case_keys is not None:
        filtered = []
        for s2p_file in s2p_files:
            variables = parse_touchstone_variables(s2p_file)
            if touchstone_key(variables) in case_keys:
                filtered.append(s2p_file)
        s2p_files = filtered
    if args.limit:
        s2p_files = s2p_files[: args.limit]

    rows = []
    worst = {name: (-np.inf, None, None, None) for name in model_names}
    for idx, s2p_file in enumerate(s2p_files, start=1):
        variables = parse_touchstone_variables(s2p_file)
        nw_hfss = rf.Network(str(s2p_file))
        length_um = variables[args.length_param]
        row = {"file": s2p_file.name, "length_um": length_um, **{name: variables[name] for name in FEATURE_ORDER}}
        pred_by_model = {}
        for model_name, meta in model_specs:
            params = predict_parameters(variables, meta)
            pred_s = calculate_s_parameters(params, length_um, nw_hfss.f)
            pred_by_model[model_name] = pred_s
            row.update({f"{model_name}_{k}": v for k, v in params.items()})
            row.update(calc_metrics(pred_s, nw_hfss.s, f"{model_name}_vs_hfss"))
            mse = row[f"{model_name}_vs_hfss_complex_mse"]
            if mse > worst[model_name][0]:
                worst[model_name] = (mse, s2p_file.name, nw_hfss, pred_by_model.copy())
        rows.append(row)
        if not args.no_plots and args.plot_first and idx <= args.plot_first:
            save_case_plot(plots_dir / f"{s2p_file.stem}_comparison.png", nw_hfss, pred_by_model, s2p_file.name)
        if idx % args.progress_every == 0:
            print(f"已处理 {idx}/{len(s2p_files)}: {s2p_file.name}")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "rdl_bottom_model_compare_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    aggregate_rows = []
    metric_cols = [c for c in summary_df.columns if c.endswith(("_mse", "_rmse", "_mae", "_db_mae", "_db_max", "_phase_mae_deg"))]
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
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_csv = out_dir / "rdl_bottom_model_compare_aggregate.csv"
    aggregate_df.to_csv(aggregate_csv, index=False)

    compact = []
    for model in model_names:
        compact.append(
            {
                "model": model,
                "mean_complex_mse": summary_df[f"{model}_vs_hfss_complex_mse"].mean(),
                "median_complex_mse": summary_df[f"{model}_vs_hfss_complex_mse"].median(),
                "max_complex_mse": summary_df[f"{model}_vs_hfss_complex_mse"].max(),
                "mean_s11_db_mae": summary_df[f"{model}_vs_hfss_s11_db_mae"].mean(),
                "mean_s21_db_mae": summary_df[f"{model}_vs_hfss_s21_db_mae"].mean(),
                "mean_s11_phase_mae_deg": summary_df[f"{model}_vs_hfss_s11_phase_mae_deg"].mean(),
                "mean_s21_phase_mae_deg": summary_df[f"{model}_vs_hfss_s21_phase_mae_deg"].mean(),
            }
        )
    compact_df = pd.DataFrame(compact).sort_values("mean_complex_mse")
    compact_csv = out_dir / "rdl_bottom_model_compare_compact.csv"
    compact_df.to_csv(compact_csv, index=False)

    if not args.no_plots and len(summary_df):
        save_summary_plots(out_dir, summary_df.reset_index(drop=True), model_names)
        for model_name, (mse, filename, nw_hfss, pred_by_model) in worst.items():
            save_case_plot(
                plots_dir / f"worst_{model_name}_{Path(filename).stem}.png",
                nw_hfss,
                pred_by_model,
                f"Worst {model_name}: {filename}, MSE={mse:.3e}",
            )

    with open(out_dir / "compare_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_cases": int(len(summary_df)),
                "models": model_names,
                "summary_csv": str(summary_csv),
                "aggregate_csv": str(aggregate_csv),
                "compact_csv": str(compact_csv),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n对比完成")
    print(f"  样本数: {len(summary_df)}")
    print(f"  模型: {', '.join(model_names)}")
    print(f"  明细: {summary_csv}")
    print(f"  汇总: {compact_csv}")
    print(compact_df.to_string(index=False))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--hfss-dir", default="data/sparameters/RDL_Bottom_Snp")
    parser.add_argument("--case-csv", default="data/tables/RDL_Bottom_TD_4.csv", help="限制对比到该 CSV 中的几何样本；留空则对比全部 s2p")
    parser.add_argument("--out-dir", default="outputs/comparison/RDL_Bottom_model_compare_new")
    parser.add_argument("--length-param", choices=["htsv", "ldown"], default="ldown")
    parser.add_argument("--mat-model", action="append", default=["mat1=data/matlab_models/RDL_TSV_mat1", "mat2=data/matlab_models/RDL_TSV_mat2"])
    parser.add_argument("--new-model-dir", default="outputs/training/RDL_Bottom_TD4_trend_sparam_training")
    parser.add_argument("--new-label", default="new_s_finetuned")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-first", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


if __name__ == "__main__":
    compare_models(build_arg_parser().parse_args())
