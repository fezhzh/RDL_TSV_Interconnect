# -*- coding: utf-8 -*-
"""Train seven shared-connection parameter networks from final optimized targets.

Run this file directly in VS Code. No command-line arguments are required.

Network requirement:
    input -> 30 -> 30 -> 20 -> 1(output)

Seven independent networks are trained, one for each Appendix-1 circuit
parameter. The predicted shared 7-parameter circuit is repeated at all 12 v11
connection positions and compared with the per-sample optimized circuit result.
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
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
OPT_RESULT_LABEL = "v11_sharedopt_c30_goodstart_all"
RUN_LABEL = "v11_shared7_param_nns_all_goodstart"

PARAM_EPOCHS = 600
PARAM_PATIENCE = 90
PARAM_LR = 6e-4
WEIGHT_DECAY = 1e-8
BATCH_SIZE = 8
PRINT_EVERY = 20
PLOT_WORST_TEST = 12
PLOT_RANDOM_TEST = 8


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


def normalize_by_train(values: np.ndarray, train_mask: np.ndarray):
    mean = values[train_mask].mean(axis=0)
    std = np.maximum(values[train_mask].std(axis=0), 1e-12)
    return (values - mean) / std, mean, std


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "train": df["split"].eq("train").to_numpy(),
        "val": df["split"].eq("val").to_numpy(),
        "test": df["split"].eq("test").to_numpy(),
    }


def train_param_model(base, wrapper, model, x_norm, y_norm, masks, device):
    train_ds = TensorDataset(
        torch.tensor(x_norm[masks["train"]], dtype=base.REAL_DTYPE),
        torch.tensor(y_norm[masks["train"]], dtype=base.REAL_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_x = torch.tensor(x_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    val_y = torch.tensor(y_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAM_LR, weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    best_val = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, PARAM_EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(val_x) - val_y) ** 2).item()
        train_loss = total / max(seen, 1)
        rows.append({"stage": "shared7_param_supervised", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[param-nn] epoch={epoch}, train={train_loss:.4e}, val={val_loss:.4e}", flush=True)
        if stale >= PARAM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def predict_params(base, model, x_norm, y_mean, y_std, device) -> np.ndarray:
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_norm), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(x_norm))
            xb = torch.tensor(x_norm[start:stop], dtype=base.REAL_DTYPE, device=device)
            pred = model(xb) * y_std_t + y_mean_t
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def evaluate_predictions(base, wrapper, dut_df, sim, opt_targets, pred_params) -> tuple[pd.DataFrame, pd.DataFrame]:
    omega = 2.0 * np.pi * sim.freq_hz
    opt_by_id = opt_targets.set_index("sample_id")
    metric_rows = []
    pred_rows = []
    for i, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        opt_p = opt_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        opt_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, opt_p))
        nn_p = pred_params[i]
        nn_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, nn_p))
        direct = metric_dict(base, target_s, direct_s)
        opt = metric_dict(base, target_s, opt_s)
        nn = metric_dict(base, target_s, nn_s)
        row = {
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
            "direct_mag_phase_mse_s11_s21": direct["mag_phase_mse_s11_s21"],
            "optimized_mag_phase_mse_s11_s21": opt["mag_phase_mse_s11_s21"],
            "nn_mag_phase_mse_s11_s21": nn["mag_phase_mse_s11_s21"],
            "direct_s11_db_mae": direct["s11_db_mae"],
            "optimized_s11_db_mae": opt["s11_db_mae"],
            "nn_s11_db_mae": nn["s11_db_mae"],
            "direct_s21_db_mae": direct["s21_db_mae"],
            "optimized_s21_db_mae": opt["s21_db_mae"],
            "nn_s21_db_mae": nn["s21_db_mae"],
        }
        metric_rows.append(row)
        pred_row = {"sample_id": sample_id, "split": sample["split"]}
        for p_idx, name in enumerate(wrapper.V08_PARAM_NAMES):
            pred_row[f"target_{name}"] = float(opt_p[p_idx])
            pred_row[f"pred_{name}"] = float(nn_p[p_idx])
        pred_rows.append(pred_row)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


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


def save_summary_plots(base, output_dir: Path, history: pd.DataFrame, metrics: pd.DataFrame, pred_params: pd.DataFrame, wrapper) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalized parameter MSE")
    axes[0].set_title("Parameter training loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].scatter(metrics["optimized_nmse_s11_s21_ri"], metrics["nn_nmse_s11_s21_ri"], s=18, alpha=0.75)
    max_nmse = float(max(metrics["optimized_nmse_s11_s21_ri"].max(), metrics["nn_nmse_s11_s21_ri"].max()))
    axes[1].plot([0, max_nmse], [0, max_nmse], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Optimized NMSE")
    axes[1].set_ylabel("NN NMSE")
    axes[1].set_title("Optimized vs NN cascade")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_and_nmse_summary.png")
    base.plt.close(fig)

    fig, axes = base.plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    for ax, name in zip(axes.ravel(), wrapper.V08_PARAM_NAMES):
        ax.scatter(pred_params[f"target_{name}"], pred_params[f"pred_{name}"], s=14, alpha=0.65)
        values = pd.concat([pred_params[f"target_{name}"], pred_params[f"pred_{name}"]]).to_numpy(dtype=np.float64)
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("optimized target")
        ax.set_ylabel("NN prediction")
        ax.grid(True, alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_prediction_scatter.png")
    base.plt.close(fig)


def plot_comparison(base, wrapper, sim, sample_idx: int, metric_row: pd.Series, opt_targets: pd.DataFrame, pred_params: np.ndarray, out_path: Path) -> None:
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    sample_id = str(metric_row["sample_id"])
    opt_p = opt_targets.set_index("sample_id").loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    target_s = sim.target_s[sample_idx]
    direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[sample_idx])))
    opt_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[sample_idx], omega, opt_p))
    nn_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[sample_idx], omega, pred_params[sample_idx]))
    fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(
        f"{sample_id} | opt={metric_row['optimized_nmse_s11_s21_ri']:.3e} | "
        f"NN={metric_row['nn_nmse_s11_s21_ri']:.3e}",
        x=0.02,
        y=0.985,
        ha="left",
    )
    specs = [
        (0, 0, "S11 real", np.real),
        (0, 0, "S11 imag", np.imag),
        (1, 0, "S21 real", np.real),
        (1, 0, "S21 imag", np.imag),
    ]
    for ax, (m, n, title, component) in zip(axes.ravel(), specs):
        ax.plot(freq_ghz, component(target_s[:, m, n]), label="HFSS", color="black", linewidth=1.8)
        ax.plot(freq_ghz, component(direct_s[:, m, n]), label="ADS direct", color="#64748b", linestyle=":")
        ax.plot(freq_ghz, component(opt_s[:, m, n]), label="Optimized params", color="#16a34a", linestyle="--")
        ax.plot(freq_ghz, component(nn_s[:, m, n]), label="NN params", color="#dc2626", linestyle="-.")
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    base.plt.close(fig)


def save_comparison_plots(base, wrapper, output_dir: Path, dut_df: pd.DataFrame, sim, metrics: pd.DataFrame, opt_targets, pred_params):
    plot_dir = output_dir / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.concat(
        [
            metrics[metrics["split"].eq("test")].sort_values("nn_nmse_s11_s21_ri", ascending=False).head(PLOT_WORST_TEST),
            metrics[metrics["split"].eq("test")].sample(
                n=min(PLOT_RANDOM_TEST, int(metrics["split"].eq("test").sum())),
                random_state=20260710,
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")
    paths = []
    for _, metric in selected.iterrows():
        idx = int(dut_df.index[dut_df["sample_id"].eq(metric["sample_id"])][0])
        out_path = plot_dir / f"{metric['sample_id']}.png"
        plot_comparison(base, wrapper, sim, idx, metric, opt_targets, pred_params, out_path)
        paths.append(str(out_path))
    return plot_dir, paths


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_shared7_nn")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_wrapper_for_shared7_nn")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_shared7_nn")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    opt_dir = version_root / "results" / OPT_RESULT_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = version_root / "results" / "v11_sharedopt_c30" / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0
    base.set_seed(base.RANDOM_SEED)

    opt_targets = pd.read_csv(opt_dir / "v08_shared_all_goodstart_targets.csv", encoding="utf-8-sig")
    dut_df = wrapper.collect_v11_samples(base)
    if list(dut_df["sample_id"]) != list(opt_targets["sample_id"]):
        opt_targets = dut_df[["sample_id"]].merge(opt_targets, on="sample_id", how="left")
    if opt_targets[wrapper.V08_PARAM_NAMES].isna().any().any():
        raise ValueError("Optimized target table is missing parameter values after sample alignment.")

    sim = base.load_single_device_simulation(dut_df, source.calibrated_ads_settings())
    masks = split_masks(opt_targets)
    x_raw = opt_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = opt_targets[wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = normalize_by_train(x_raw, masks["train"])
    y_norm, y_mean, y_std = normalize_by_train(y_raw, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    model = wrapper.SharedV08ParamNet(input_dim=x_norm.shape[1]).to(dtype=base.REAL_DTYPE, device=device)
    history = train_param_model(base, wrapper, model, x_norm, y_norm, masks, device)
    pred_params = predict_params(base, model, x_norm, y_mean, y_std, device)
    metrics, pred_table = evaluate_predictions(base, wrapper, dut_df, sim, opt_targets, pred_params)
    summary = summarize(metrics)
    save_summary_plots(base, output_dir, history, metrics, pred_table, wrapper)
    plot_dir, plot_paths = save_comparison_plots(base, wrapper, output_dir, dut_df, sim, metrics, opt_targets, pred_params)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": "v11_shared7_param_nns",
                "architecture": "seven independent input->30->30->20->1 networks",
                "feature_columns": base.STRUCTURE_COLUMNS,
                "target_columns": wrapper.V08_PARAM_NAMES,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "source_optimized_targets": str(opt_dir / "v08_shared_all_goodstart_targets.csv"),
                "connection_count": wrapper.CONNECTION_COUNT,
                "v08_scale_factors": wrapper.V08_SCALE_FACTORS.tolist(),
                "freq_hz": sim.freq_hz.tolist(),
            },
        },
        output_dir / "shared7_param_nns.pt",
    )

    history.to_csv(output_dir / "shared7_param_training_history.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(output_dir / "shared7_param_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "optimized_vs_shared7_nn_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "optimized_vs_shared7_nn_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "source_optimized_targets": str(opt_dir / "v08_shared_all_goodstart_targets.csv"),
        "architecture": "seven independent input->30->30->20->1 networks",
        "samples": int(len(dut_df)),
        "epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
        "best_val_loss": float(history["val_loss"].min()) if len(history) else None,
        "summary": summary.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_paths,
    }
    (output_dir / "shared7_param_nn_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "shared7_param_nn_report.md").write_text(
        "\n".join(
            [
                "# V11 Shared 7-Parameter NN Report",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source optimized targets: `{opt_dir / 'v08_shared_all_goodstart_targets.csv'}`",
                "- Network: seven independent `input -> 30 -> 30 -> 20 -> 1` models, one per circuit parameter.",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Best validation loss: `{report['best_val_loss']}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Outputs",
                "",
                f"- Checkpoint: `{output_dir / 'shared7_param_nns.pt'}`",
                f"- Training history: `{output_dir / 'shared7_param_training_history.csv'}`",
                f"- Parameter predictions: `{output_dir / 'shared7_param_predictions.csv'}`",
                f"- Metrics: `{output_dir / 'optimized_vs_shared7_nn_metrics.csv'}`",
                f"- Summary: `{output_dir / 'optimized_vs_shared7_nn_summary.csv'}`",
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
                f"- Train/val/test: `{int(masks['train'].sum())}` / `{int(masks['val'].sum())}` / `{int(masks['test'].sum())}`",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Comparison plots: `{len(plot_paths)}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
