import argparse
import faulthandler
import re
from pathlib import Path

faulthandler.enable()

import numpy as np
import pandas as pd
import scipy.io as sio
import skrf as rf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
FEATURE_ORDER = ["ldown", "wdown", "tdown", "htsv", "p1"]


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


def load_model_meta(mat_dir, prefix="RDL_Bottom_"):
    dims = {}
    for name in TARGET_PARAMS:
        mat_path = Path(mat_dir) / f"{prefix}{name}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {mat_path}")
        data = sio.loadmat(mat_path)
        dims[name] = int(np.asarray(data["psmin"]).size)

    unique_dims = sorted(set(dims.values()))
    if len(unique_dims) != 1:
        raise ValueError(f"{mat_dir} 中模型输入维度不一致: {dims}")
    dim = unique_dims[0]
    if dim not in (3, 5):
        raise ValueError(f"{mat_dir} 中模型输入维度为 {dim}，脚本只支持 3 或 5")

    return {
        "dir": Path(mat_dir),
        "prefix": prefix,
        "input_dim": dim,
        "feature_names": FEATURE_ORDER[:dim],
    }


def predict_circuit_parameters(features, model_meta):
    x = np.asarray(features, dtype=float).reshape(1, -1)
    if x.shape[1] != model_meta["input_dim"]:
        raise ValueError(
            f"{model_meta['dir']} 需要 {model_meta['input_dim']} 个输入，实际得到 {x.shape[1]} 个"
        )

    circuit_params = {}
    for name in TARGET_PARAMS:
        mat_path = model_meta["dir"] / f"{model_meta['prefix']}{name}.mat"
        data = sio.loadmat(mat_path)

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


def calculate_s_parameters(circuit_params, length_um, freqs_hz):
    R1 = circuit_params["R1"]
    R2 = circuit_params["R2"]
    R3 = circuit_params["R3"]
    L1 = circuit_params["L1"] * 1e-9
    L2 = circuit_params["L2"] * 1e-9
    L3 = circuit_params["L3"] * 1e-9
    Cox = circuit_params["Cox"] * 1e-12
    Csi = circuit_params["Csi"] * 1e-12
    Rsi = circuit_params["Rsi"]

    length_m = length_um * 1e-6
    omega = 2.0 * np.pi * freqs_hz

    R_rlgc = (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2) / (
        (R1 + R2) ** 2 + omega**2 * L2**2
    ) + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    L_rlgc = (R1**2 * L2) / ((R1 + R2) ** 2 + omega**2 * L2**2) + (
        L3 * R3**2
    ) / (R3**2 + omega**2 * L3**2) + L1
    G_rlgc = (omega**2 * Rsi * Cox**2) / (
        1.0 + omega**2 * Rsi**2 * (Cox + Csi) ** 2
    )
    C_rlgc = (
        Cox + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)
    ) / (1.0 + omega**2 * Rsi**2 * (Cox + Csi) ** 2)

    z0 = np.sqrt((R_rlgc + 1j * omega * L_rlgc) / (G_rlgc + 1j * omega * C_rlgc))
    gamma = np.sqrt((R_rlgc + 1j * omega * L_rlgc) * (G_rlgc + 1j * omega * C_rlgc))

    A = np.cosh(gamma * length_m)
    B = z0 * np.sinh(gamma * length_m)
    C = np.sinh(gamma * length_m) / z0
    D = np.cosh(gamma * length_m)

    denom = A + B / 50.0 + C * 50.0 + D
    s = np.empty((len(freqs_hz), 2, 2), dtype=complex)
    s[:, 0, 0] = (A + B / 50.0 - C * 50.0 - D) / denom
    s[:, 0, 1] = 2.0 * (A * D - B * C) / denom
    s[:, 1, 0] = 2.0 / denom
    s[:, 1, 1] = (-A + B / 50.0 - C * 50.0 + D) / denom
    return s


def db20(value):
    return 20.0 * np.log10(np.maximum(np.abs(value), 1e-300))


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


def configure_matplotlib():
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f8fb",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1f2937",
            "axes.labelcolor": "#1f2937",
            "axes.titlecolor": "#111827",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cbd5e1",
            "savefig.facecolor": "#f6f8fb",
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, pad=8)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)


def save_case_plot(out_path, nw_hfss, pred_by_model, title):
    configure_matplotlib()
    freq_ghz = nw_hfss.f / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=16, fontweight="semibold")

    summary_parts = []
    for model_name, pred_s in pred_by_model.items():
        mse = np.mean(np.abs(pred_s - nw_hfss.s) ** 2)
        s21_mae = np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(nw_hfss.s[:, 1, 0])))
        summary_parts.append(f"{model_name}: MSE {mse:.3e}, S21 MAE {s21_mae:.3f} dB")
    fig.text(0.02, 0.953, "    ".join(summary_parts), ha="left", va="top", fontsize=10, color="#475569")

    ports = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]
    model_colors = {"mat2": "#dc2626", "mat3": "#059669", "mat4": "#7e22ce"}

    for ax, (m, n, name) in zip(axes.ravel(), ports):
        ax.plot(freq_ghz, db20(nw_hfss.s[:, m, n]), label="HFSS", color="#1f77b4", linewidth=1.8)
        for model_name, pred_s in pred_by_model.items():
            ax.plot(
                freq_ghz,
                db20(pred_s[:, m, n]),
                label=model_name,
                color=model_colors.get(model_name, "#7e22ce"),
                linestyle="--",
                linewidth=1.8,
            )
        style_axis(ax, f"{name} magnitude", "Magnitude (dB)")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.91, wspace=0.18, hspace=0.28)
    fig.savefig(out_path)
    plt.close(fig)


def save_summary_plots(out_dir, summary_df, model_names):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.suptitle("RDL_Bottom Model Error Summary", x=0.02, y=0.98, ha="left", fontsize=16, fontweight="semibold")
    fig.text(
        0.02,
        0.925,
        f"Valid cases: {len(summary_df)}    Lower values indicate closer agreement with HFSS",
        ha="left",
        fontsize=10,
        color="#475569",
    )
    x = np.arange(len(summary_df))
    colors = {"mat2": "#dc2626", "mat3": "#059669", "mat4": "#7e22ce"}

    for model in model_names:
        axes[0].plot(
            x,
            summary_df[f"{model}_vs_hfss_complex_mse"].to_numpy(dtype=float),
            label=model,
            color=colors.get(model, "#1f77b4"),
            linewidth=1.5,
        )
        axes[1].plot(
            x,
            summary_df[f"{model}_vs_hfss_s21_db_mae"].to_numpy(dtype=float),
            label=model,
            color=colors.get(model, "#1f77b4"),
            linewidth=1.5,
        )

    axes[0].set_yscale("log")
    axes[0].set_title("Complex S MSE vs HFSS", loc="left", fontsize=12, pad=8)
    axes[0].set_xlabel("Case index")
    axes[0].set_ylabel("MSE")
    axes[1].set_title("S21 magnitude MAE vs HFSS", loc="left", fontsize=12, pad=8)
    axes[1].set_xlabel("Case index")
    axes[1].set_ylabel("MAE (dB)")

    for ax in axes:
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.84, wspace=0.18)
    fig.savefig(out_dir / "summary_error_trends.png")
    plt.close(fig)


def compare_models(args):
    if args.base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
    else:
        base_dir = Path(args.base_dir).resolve()
    hfss_dir = (base_dir / args.hfss_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("mat2", load_model_meta(base_dir / "data" / "matlab_models" / "RDL_TSV_mat2")),
        ("mat3", load_model_meta(base_dir / "data" / "matlab_models" / "RDL_TSV_mat3")),
        ("mat4", load_model_meta(base_dir / "data" / "matlab_models" / "RDL_TSV_mat4")),
    ]

    s2p_files = sorted(hfss_dir.glob("*.s2p"), key=natural_key)
    if args.limit:
        s2p_files = s2p_files[: args.limit]
    if not s2p_files:
        raise FileNotFoundError(f"{hfss_dir} 中未找到 .s2p 文件")

    rows = []
    worst = {name: (-np.inf, None, None, None) for name, _ in model_specs}
    model_names = [name for name, _ in model_specs]

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
            params_by_model = {}
            for model_name, meta in model_specs:
                features = np.array([variables[name] for name in meta["feature_names"]], dtype=float)
                circuit_params = predict_circuit_parameters(features, meta)
                pred_s = calculate_s_parameters(circuit_params, length_um, nw_hfss.f)

                pred_by_model[model_name] = pred_s
                params_by_model[model_name] = circuit_params
                row[f"{model_name}_features"] = ",".join(meta["feature_names"])
                row.update({f"{model_name}_{k}": v for k, v in circuit_params.items()})
                row.update(calc_metrics(pred_s, nw_hfss.s, f"{model_name}_vs_hfss"))

                mse = row[f"{model_name}_vs_hfss_complex_mse"]
                if mse > worst[model_name][0]:
                    worst[model_name] = (mse, s2p_file.name, nw_hfss, pred_by_model.copy())

            if "mat2" in pred_by_model:
                for compare_name, compare_s in pred_by_model.items():
                    if compare_name == "mat2":
                        continue
                    row.update(calc_metrics(compare_s, pred_by_model["mat2"], f"{compare_name}_vs_mat2"))
            rows.append(row)

            if not args.no_plots and (args.plot_all or (args.plot_first and idx <= args.plot_first)):
                save_case_plot(
                    plots_dir / f"{s2p_file.stem}_comparison.png",
                    nw_hfss,
                    pred_by_model,
                    f"{s2p_file.name} RDL_Bottom model comparison",
                )

            if idx % args.progress_every == 0:
                print(f"已处理 {idx}/{len(s2p_files)}: {s2p_file.name}")
        except Exception as exc:
            rows.append({"file": s2p_file.name, "error": str(exc)})
            print(f"[跳过] {s2p_file.name}: {exc}")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "rdl_bottom_model_compare_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    valid = summary_df[summary_df.get("error").isna()] if "error" in summary_df else summary_df
    aggregate_rows = []
    metric_cols = [
        col
        for col in valid.columns
        if col.endswith("_mse")
        or col.endswith("_rmse")
        or col.endswith("_mae")
        or col.endswith("_db_mae")
        or col.endswith("_db_max")
        or col.endswith("_phase_mae_deg")
    ]
    for col in metric_cols:
        aggregate_rows.append(
            {
                "metric": col,
                "mean": valid[col].mean(),
                "median": valid[col].median(),
                "min": valid[col].min(),
                "max": valid[col].max(),
                "worst_file": valid.loc[valid[col].idxmax(), "file"] if len(valid) else None,
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_csv = out_dir / "rdl_bottom_model_compare_aggregate.csv"
    aggregate_df.to_csv(aggregate_csv, index=False)

    if not args.no_plots and len(valid):
        save_summary_plots(out_dir, valid.reset_index(drop=True), model_names)
        for model_name, (mse, filename, nw_hfss, pred_by_model) in worst.items():
            if filename is not None:
                save_case_plot(
                    plots_dir / f"worst_{model_name}_{Path(filename).stem}.png",
                    nw_hfss,
                    pred_by_model,
                    f"Worst {model_name} vs HFSS: {filename}, MSE={mse:.3e}",
                )

    print("\n对比完成")
    print(f"  HFSS 文件夹: {hfss_dir}")
    print(f"  有效样本数: {len(valid)} / {len(summary_df)}")
    print(f"  明细 CSV: {summary_csv}")
    print(f"  汇总 CSV: {aggregate_csv}")
    if not args.no_plots:
        print(f"  图像目录: {plots_dir}")
    else:
        print("  PNG 绘图: 已关闭")

    if len(valid):
        for model_name in model_names:
            mse_col = f"{model_name}_vs_hfss_complex_mse"
            s21_col = f"{model_name}_vs_hfss_s21_db_mae"
            print(
                f"  {model_name}: mean MSE={valid[mse_col].mean():.6e}, "
                f"mean S21 dB MAE={valid[s21_col].mean():.6f} dB"
            )
        for compare_name in model_names:
            if compare_name == "mat2":
                continue
            mse_col = f"{compare_name}_vs_mat2_complex_mse"
            s21_col = f"{compare_name}_vs_mat2_s21_db_mae"
            if mse_col in valid.columns and s21_col in valid.columns:
                print(
                    f"  {compare_name} vs mat2: mean MSE={valid[mse_col].mean():.6e}, "
                    f"mean S21 dB MAE={valid[s21_col].mean():.6f} dB"
                )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare RDL_Bottom predictions from RDL_TSV_mat2, RDL_TSV_mat3, and RDL_TSV_mat4 against HFSS s2p files."
    )
    parser.add_argument("--base-dir", default=None, help="工程根目录，默认自动定位到项目根目录")
    parser.add_argument("--hfss-dir", default="data/sparameters/RDL_Bottom_Snp", help="HFSS RDL_Bottom s2p 文件夹")
    parser.add_argument(
        "--out-dir",
        default="outputs/comparison/RDL_Bottom_model_compare",
        help="输出目录，包含 CSV 和 PNG",
    )
    parser.add_argument(
        "--length-param",
        choices=["htsv", "ldown"],
        default="ldown",
        help="RLGC 传输线长度来源。默认 ldown；可切换 htsv。",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件，调试用")
    parser.add_argument("--no-plots", action="store_true", help="不生成 PNG 对比图，只导出 CSV")
    parser.add_argument("--plot-all", action="store_true", help="保存每一个 dut 的对比图，可能生成较多 PNG 文件")
    parser.add_argument("--plot-first", type=int, default=3, help="额外绘制前 N 个样本的对比图")
    parser.add_argument("--progress-every", type=int, default=25, help="每处理 N 个文件打印一次进度")
    return parser


if __name__ == "__main__":
    compare_models(build_arg_parser().parse_args())
