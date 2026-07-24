# -*- coding: utf-8 -*-
"""Train a DUT-level NN for optimized connection-network scale parameters.

Run this file directly in VS Code after ``Calc_SP_and_Opt2.py`` has generated
``connection_network_params.csv``. One training sample is one DUT structure:
the input is the DUT geometry, and the output is all 8 connection networks'
scale parameters.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skrf as rf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPARAM_SCRIPT_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "code"
if str(SPARAM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SPARAM_SCRIPT_DIR))

import Calc_SP_and_Opt2 as opt2


INPUT_CSV = (
    PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "results" / "RDL_TSV_mat4_opt2"
    / "connection_network_params.csv"
)
# Configure these values before running directly from VS Code.
TARGET_VARIANT = "optimized_with_cn3"  # "optimized_with_cn3" or "optimized_without_cn3"
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v07_connection_param_and_sparam_finetune" / "results" / f"connection_network_param_model_{TARGET_VARIANT}"
RANDOM_SEED = 20260627
TRAIN_RATIO = 0.8
BATCH_SIZE = 128
EPOCHS = 4000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_LAYERS = [256, 256, 128, 128]
PATIENCE = 350
PRINT_EVERY = 100
USE_CUDA_IF_AVAILABLE = True
WRITE_STRUCTURE_PLOTS = True
PLOT_SPLIT = "val"  # "val", "train", or "all"
PLOT_DUT_LIMIT = None

STRUCTURE_COLUMNS = [
    "structure_lrdl",
    "structure_wrdl",
    "structure_trdl",
    "structure_ldown",
    "structure_wdown",
    "structure_tdown",
    "structure_dtsv",
    "structure_htsv",
    "structure_p1",
]
CONNECTION_COUNT = 8
SCALE_COLUMNS_BY_VARIANT = {
    "optimized_with_cn3": ["Cn1_scale", "Rn1_scale", "Cn2_scale", "Rn2_scale", "Cn3_scale", "Rn3_scale", "Ln1_scale"],
    "optimized_without_cn3": ["Cn1_scale", "Rn1_scale", "Cn2_scale", "Rn2_scale", "Rn3_scale", "Ln1_scale"],
}
if TARGET_VARIANT not in SCALE_COLUMNS_BY_VARIANT:
    raise ValueError(f"不支持的 TARGET_VARIANT: {TARGET_VARIANT}")
SCALE_COLUMNS = SCALE_COLUMNS_BY_VARIANT[TARGET_VARIANT]
INCLUDE_CN3_IN_CASCADE = TARGET_VARIANT == "optimized_with_cn3"
TARGET_COLUMNS = [
    f"conn{conn_idx}_{name}"
    for conn_idx in range(1, CONNECTION_COUNT + 1)
    for name in SCALE_COLUMNS
]


class DutConnectionParamNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden_dim))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_connection_dataframe():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"未找到训练数据: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = ["file", "dut_index", "variant", "connection_index"] + STRUCTURE_COLUMNS + SCALE_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"训练数据缺少列: {missing}")

    df = df[df["variant"] == TARGET_VARIANT].copy()
    if df.empty:
        raise ValueError(f"没有找到 variant={TARGET_VARIANT!r} 的训练数据")

    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("清理缺失值后没有可训练数据")

    return df


def build_dut_dataframe(connection_df):
    rows = []
    for dut_idx, group in connection_df.groupby("dut_index", sort=True):
        group = group.sort_values("connection_index")
        if len(group) != CONNECTION_COUNT:
            continue
        expected = list(range(1, CONNECTION_COUNT + 1))
        actual = group["connection_index"].astype(int).tolist()
        if actual != expected:
            continue

        first = group.iloc[0]
        row = {
            "file": first["file"],
            "dut_index": int(dut_idx),
            "variant": TARGET_VARIANT,
        }
        for col in STRUCTURE_COLUMNS:
            row[col] = float(first[col])
        for _, conn_row in group.iterrows():
            conn_idx = int(conn_row["connection_index"])
            for name in SCALE_COLUMNS:
                row[f"conn{conn_idx}_{name}"] = float(conn_row[name])
        rows.append(row)

    dut_df = pd.DataFrame(rows)
    if dut_df.empty:
        raise ValueError("没有完整的 8 连接 DUT 训练样本")
    return dut_df


def split_by_dut(dut_df):
    dut_ids = dut_df["dut_index"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = dut_ids.copy()
    rng.shuffle(shuffled)

    if len(shuffled) < 2:
        mask = rng.random(len(dut_df)) < TRAIN_RATIO
        return mask, ~mask

    n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * TRAIN_RATIO))))
    train_duts = set(shuffled[:n_train].tolist())
    train_mask = dut_df["dut_index"].isin(train_duts).to_numpy()
    return train_mask, ~train_mask


def normalize_train_val(values, train_mask):
    mean = values[train_mask].mean(axis=0)
    std = np.maximum(values[train_mask].std(axis=0), 1e-12)
    return (values - mean) / std, mean, std


def build_matrices(dut_df):
    x_raw = dut_df[STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = dut_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    train_mask, val_mask = split_by_dut(dut_df)

    x_norm, x_mean, x_std = normalize_train_val(x_raw, train_mask)
    y_norm, y_mean, y_std = normalize_train_val(y_raw, train_mask)

    metadata = {
        "feature_columns": STRUCTURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "scale_columns": SCALE_COLUMNS,
        "connection_count": CONNECTION_COUNT,
        "target_variant": TARGET_VARIANT,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    return x_norm, y_norm, y_raw, train_mask, val_mask, metadata


def evaluate(model, x_tensor, y_tensor, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        pred_norm = model(x_tensor)
        mse_norm = torch.mean((pred_norm - y_tensor) ** 2).item()
        pred = pred_norm.cpu().numpy() * y_std + y_mean
        target = y_tensor.cpu().numpy() * y_std + y_mean
        mae = np.mean(np.abs(pred - target), axis=0)
        rmse = np.sqrt(np.mean((pred - target) ** 2, axis=0))
    return mse_norm, mae, rmse, pred


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def plot_structure_case(idx, split_name, hfss_nw, direct_s, optimized_s, nn_pred_s, out_path):
    freq_ghz = hfss_nw.f / 1e9
    ports = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    fig.suptitle(
        f"dut{idx}.s2p {TARGET_VARIANT} NN DUT-level prediction ({split_name})",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.02,
        0.955,
        (
            f"Direct MSE={opt2.mse(hfss_nw.s, direct_s):.3e} | "
            f"Optimized target MSE={opt2.mse(hfss_nw.s, optimized_s):.3e} | "
            f"NN predicted MSE={opt2.mse(hfss_nw.s, nn_pred_s):.3e}"
        ),
        ha="left",
        va="top",
        fontsize=9,
        color="#475569",
    )

    for ax, (m, n, label) in zip(axes.ravel(), ports):
        ax.plot(freq_ghz, db20(hfss_nw.s[:, m, n]), label="HFSS", color="black", linewidth=2.0)
        ax.plot(freq_ghz, db20(direct_s[:, m, n]), label="Direct cascade", color="#64748b", linestyle=":", linewidth=1.7)
        ax.plot(freq_ghz, db20(optimized_s[:, m, n]), label="Optimized target", color="#dc2626", linestyle="--", linewidth=1.7)
        ax.plot(freq_ghz, db20(nn_pred_s[:, m, n]), label="NN predicted cascade", color="#16a34a", linestyle="-.", linewidth=1.7)
        ax.set_title(f"{label} magnitude")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.9, wspace=0.2, hspace=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_structure_prediction_outputs(pred_df):
    plot_df = pred_df.copy()
    if PLOT_SPLIT != "all":
        plot_df = plot_df[plot_df["split"] == PLOT_SPLIT].copy()
    if plot_df.empty:
        return pd.DataFrame()

    plot_dir = OUTPUT_DIR / "structure_prediction_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    n_plotted = 0
    for _, row in plot_df.sort_values("dut_index").iterrows():
        if PLOT_DUT_LIMIT is not None and n_plotted >= PLOT_DUT_LIMIT:
            break

        idx = int(row["dut_index"])
        hfss_path = opt2.S2P_DIR / f"dut{idx}.s2p"
        optimized_path = opt2.OUTPUT_DIR / TARGET_VARIANT / f"dut{idx}.s2p"
        if not hfss_path.exists() or not optimized_path.exists():
            continue

        hfss_nw = rf.Network(str(hfss_path))
        optimized_nw = rf.Network(str(optimized_path))
        params = opt2.extract_device_params_RDL_TSV(hfss_path)
        base_abcds = opt2.build_base_abcds(params, hfss_nw.f)
        direct_s = opt2.abcd2s(opt2.cascade_direct(base_abcds))

        p_all = []
        for conn_idx in range(1, CONNECTION_COUNT + 1):
            for name in SCALE_COLUMNS:
                p_all.append(row[f"pred_conn{conn_idx}_{name}"])

        nn_pred_s = opt2.abcd2s(
            opt2.cascade_with_corrections(
                base_abcds,
                2 * np.pi * hfss_nw.f,
                np.array(p_all),
                include_cn3=INCLUDE_CN3_IN_CASCADE,
            )
        )

        split_name = row["split"]
        out_path = plot_dir / f"dut{idx}_{TARGET_VARIANT}_{split_name}_structure_prediction.png"
        plot_structure_case(idx, split_name, hfss_nw, direct_s, optimized_nw.s, nn_pred_s, out_path)

        metric_rows.append(
            {
                "dut_index": idx,
                "variant": TARGET_VARIANT,
                "split": split_name,
                "plot_file": str(out_path),
                "direct_mse_vs_hfss": opt2.mse(hfss_nw.s, direct_s),
                "optimized_mse_vs_hfss": opt2.mse(hfss_nw.s, optimized_nw.s),
                "nn_pred_mse_vs_hfss": opt2.mse(hfss_nw.s, nn_pred_s),
                "nn_pred_mse_vs_optimized": opt2.mse(optimized_nw.s, nn_pred_s),
            }
        )
        n_plotted += 1

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        metrics_df.to_csv(OUTPUT_DIR / "structure_prediction_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics_df


def train_model(x_norm, y_norm, train_mask, val_mask):
    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    x_train = torch.tensor(x_norm[train_mask], dtype=torch.float64)
    y_train = torch.tensor(y_norm[train_mask], dtype=torch.float64)
    x_val = torch.tensor(x_norm[val_mask], dtype=torch.float64, device=device)
    y_val = torch.tensor(y_norm[val_mask], dtype=torch.float64, device=device)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    model = DutConnectionParamNet(x_norm.shape[1], y_norm.shape[1], HIDDEN_LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)

    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            n_seen += len(xb)

        train_loss = train_loss_sum / max(n_seen, 1)
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(x_val) - y_val) ** 2).item() if len(x_val) else train_loss
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"epoch={epoch}, train_loss={train_loss:.6e}, val_loss={val_loss:.6e}")
        if stale_epochs >= PATIENCE:
            print(f"早停: epoch={epoch}, best_val_loss={best_val:.6e}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history), device


def save_outputs(model, history_df, dut_df, x_norm, y_norm, y_raw, train_mask, val_mask, metadata, device):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "hidden_layers": HIDDEN_LAYERS,
    }
    torch.save(checkpoint, OUTPUT_DIR / "connection_param_net.pt")

    with open(OUTPUT_DIR / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    x_all = torch.tensor(x_norm, dtype=torch.float64, device=device)
    y_all = torch.tensor(y_norm, dtype=torch.float64, device=device)
    _, mae_all, rmse_all, pred_all = evaluate(
        model,
        x_all,
        y_all,
        np.array(metadata["y_mean"], dtype=np.float64),
        np.array(metadata["y_std"], dtype=np.float64),
    )

    pred_df = dut_df[["file", "dut_index", "variant"] + STRUCTURE_COLUMNS].copy()
    pred_df["split"] = np.where(train_mask, "train", "val")
    for i, col in enumerate(TARGET_COLUMNS):
        pred_df[f"target_{col}"] = y_raw[:, i]
        pred_df[f"pred_{col}"] = pred_all[:, i]
        pred_df[f"abs_error_{col}"] = np.abs(pred_all[:, i] - y_raw[:, i])
    pred_df.to_csv(OUTPUT_DIR / "connection_param_predictions.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    for i, col in enumerate(TARGET_COLUMNS):
        metric_rows.append({"target": col, "mae_all": float(mae_all[i]), "rmse_all": float(rmse_all[i])})
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")

    structure_metrics_df = pd.DataFrame()
    if WRITE_STRUCTURE_PLOTS:
        structure_metrics_df = save_structure_prediction_outputs(pred_df)

    report = {
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUTPUT_DIR),
        "n_rows": int(len(dut_df)),
        "n_train_rows": int(train_mask.sum()),
        "n_val_rows": int(val_mask.sum()),
        "n_dut": int(dut_df["dut_index"].nunique()),
        "target_variant": TARGET_VARIANT,
        "target_layout": f"one_dut_to_8_connections_{len(TARGET_COLUMNS)}_scales",
        "include_cn3_in_cascade": INCLUDE_CN3_IN_CASCADE,
        "device": str(device),
        "final_epoch": int(history_df["epoch"].iloc[-1]) if len(history_df) else 0,
        "best_val_loss": float(history_df["val_loss"].min()) if len(history_df) else None,
        "n_structure_prediction_plots": int(len(structure_metrics_df)),
        "structure_prediction_metrics_csv": str(OUTPUT_DIR / "structure_prediction_metrics.csv")
        if len(structure_metrics_df)
        else None,
    }
    with open(OUTPUT_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return metrics_df, report


def main():
    set_seed(RANDOM_SEED)
    connection_df = load_connection_dataframe()
    dut_df = build_dut_dataframe(connection_df)
    x_norm, y_norm, y_raw, train_mask, val_mask, metadata = build_matrices(dut_df)

    print(f"训练数据: {INPUT_CSV}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"DUT 样本数: {len(dut_df)}")
    print(f"训练 DUT 数: {train_mask.sum()}, 验证 DUT 数: {val_mask.sum()}")
    print(f"输入维度: {x_norm.shape[1]}, 输出维度: {y_norm.shape[1]}")

    model, history_df, device = train_model(x_norm, y_norm, train_mask, val_mask)
    metrics_df, report = save_outputs(model, history_df, dut_df, x_norm, y_norm, y_raw, train_mask, val_mask, metadata, device)

    print("\n训练完成")
    print(f"模型文件: {OUTPUT_DIR / 'connection_param_net.pt'}")
    print(f"预测对比: {OUTPUT_DIR / 'connection_param_predictions.csv'}")
    if report["structure_prediction_metrics_csv"]:
        print(f"整体结构预测指标: {report['structure_prediction_metrics_csv']}")
        print(f"整体结构对比图目录: {OUTPUT_DIR / 'structure_prediction_plots'}")
    print(f"最佳验证 loss: {report['best_val_loss']:.6e}")
    print(metrics_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
