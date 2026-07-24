# -*- coding: utf-8 -*-
"""Fine-tune MATLAB RDL parameter NNs with complex S-parameter loss.

Run directly in VS Code after:
1. extract_rdl_params_for_lhs_dataset_comparison.py
2. nn_train_3.m
"""

import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio
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
PARAM_TABLE_ROOT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "results" / "extracted_params"
MATLAB_MODEL_ROOT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "models" / "matlab_param_nns"
OUTPUT_ROOT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "results" / "sparam_finetuned_models"

DATASET_NAMES = ["lhs100", "lhs200", "lhs400", "lhs800", "lhs100_lhs200_lhs400_lhs800"]
DEVICE_NAMES = ["TMRDL", "BSMRDL"]
TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
DEVICE_CONFIGS = {
    "TMRDL": {
        "features": ["pitch", "l_tmrdl", "w_tmrdl", "h_tmrdl"],
        "length_column": "l_tmrdl",
    },
    "BSMRDL": {
        "features": ["pitch", "l_bsmrdl", "w_bsmrdl", "h_bsmrdl"],
        "length_column": "l_bsmrdl",
    },
}

RANDOM_SEED = 20260706
USE_CUDA_IF_AVAILABLE = True
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
Z_REF = 50.0

SPARAM_EPOCHS = 1200
SPARAM_PATIENCE = 180
SPARAM_BATCH_SIZE = 16
SPARAM_LR = 2e-6
SPARAM_WEIGHT_DECAY = 1e-10
PRINT_EVERY = 50
PLOT_WORST_N = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MatlabSingleParamNet(nn.Module):
    def __init__(self, mat_path):
        super().__init__()
        data = sio.loadmat(mat_path)
        self.w1 = nn.Parameter(torch.tensor(data["w1"], dtype=REAL_DTYPE))
        self.theta1 = nn.Parameter(torch.tensor(data["theta1"], dtype=REAL_DTYPE))
        self.w2 = nn.Parameter(torch.tensor(data["w2"], dtype=REAL_DTYPE))
        self.theta2 = nn.Parameter(torch.tensor(data["theta2"], dtype=REAL_DTYPE))
        self.w3 = nn.Parameter(torch.tensor(data["w3"], dtype=REAL_DTYPE))
        self.theta3 = nn.Parameter(torch.tensor(data["theta3"], dtype=REAL_DTYPE))
        self.register_buffer("psmin", torch.tensor(data["psmin"], dtype=REAL_DTYPE).reshape(1, -1))
        self.register_buffer("psmax", torch.tensor(data["psmax"], dtype=REAL_DTYPE).reshape(1, -1))
        self.register_buffer("outputmin", torch.tensor(data["outputmin"], dtype=REAL_DTYPE).reshape(1, 1))
        self.register_buffer("outputmax", torch.tensor(data["outputmax"], dtype=REAL_DTYPE).reshape(1, 1))

    def forward(self, x_raw):
        denom = torch.clamp(self.psmax - self.psmin, min=1e-30)
        x = 2.0 * (x_raw - self.psmin) / denom - 1.0
        y = torch.tanh(x @ self.w1 + self.theta1)
        y = torch.tanh(y @ self.w2 + self.theta2)
        y = y @ self.w3 + self.theta3
        out = self.outputmin + (y + 1.0) * (self.outputmax - self.outputmin) / 2.0
        return out.squeeze(-1)


class MatlabMultiParamNet(nn.Module):
    def __init__(self, dataset_name, device_name):
        super().__init__()
        model_dir = MATLAB_MODEL_ROOT / dataset_name
        self.param_nets = nn.ModuleList(
            [MatlabSingleParamNet(model_dir / f"{device_name}_{name}.mat") for name in TARGET_PARAMS]
        )

    def forward(self, x_raw):
        return torch.stack([net(x_raw) for net in self.param_nets], dim=1)


def split_masks(df):
    return {name: (df["split"].to_numpy() == name) for name in ["train", "val", "test"]}


def load_param_table(dataset_name, device_name):
    csv_path = PARAM_TABLE_ROOT / dataset_name / f"{device_name}_circuit_params.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing extracted parameter table: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def load_s_targets(df):
    s_rows = []
    freq = None
    for path in df["snp_path"]:
        nw = rf.Network(str(path))
        if freq is None:
            freq = nw.f
        if len(freq) != len(nw.f) or not np.allclose(freq, nw.f):
            raise ValueError(f"Frequency grid mismatch: {path}")
        s_rows.append(nw.s)
    return np.stack(s_rows, axis=0), freq


def circuit_params_to_s_torch(params, length_um, freqs_hz):
    r1, r2, r3 = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    l1, l2, l3 = params[:, 3:4] * 1e-9, params[:, 4:5] * 1e-9, params[:, 5:6] * 1e-9
    cox, csi, rsi = params[:, 6:7] * 1e-12, params[:, 7:8] * 1e-12, params[:, 8:9]
    omega = torch.tensor(2.0 * np.pi * freqs_hz, dtype=REAL_DTYPE, device=params.device)[None, :]
    length_m = length_um[:, None].to(params.device) * 1e-6
    j = torch.complex(
        torch.tensor(0.0, dtype=REAL_DTYPE, device=params.device),
        torch.tensor(1.0, dtype=REAL_DTYPE, device=params.device),
    )
    r_rlgc = (r1**2 * r2 + r1 * r2**2 + omega**2 * r1 * l2**2) / (
        (r1 + r2) ** 2 + omega**2 * l2**2 + 1e-30
    ) + (omega**2 * l3**2 * r3) / (r3**2 + omega**2 * l3**2 + 1e-30)
    l_rlgc = (r1**2 * l2) / ((r1 + r2) ** 2 + omega**2 * l2**2 + 1e-30) + l3 * r3**2 / (
        r3**2 + omega**2 * l3**2 + 1e-30
    ) + l1
    g_rlgc = (omega**2 * rsi * cox**2) / (1.0 + omega**2 * rsi**2 * (cox + csi) ** 2 + 1e-30)
    c_rlgc = (cox + omega**2 * csi * rsi**2 * cox * (cox + csi)) / (
        1.0 + omega**2 * rsi**2 * (cox + csi) ** 2 + 1e-30
    )
    z0 = torch.sqrt(
        (r_rlgc.to(COMPLEX_DTYPE) + j * omega * l_rlgc.to(COMPLEX_DTYPE))
        / (g_rlgc.to(COMPLEX_DTYPE) + j * omega * c_rlgc.to(COMPLEX_DTYPE) + 1e-300)
    )
    gamma = torch.sqrt(
        (r_rlgc.to(COMPLEX_DTYPE) + j * omega * l_rlgc.to(COMPLEX_DTYPE))
        * (g_rlgc.to(COMPLEX_DTYPE) + j * omega * c_rlgc.to(COMPLEX_DTYPE))
    )
    gl = gamma * length_m.to(COMPLEX_DTYPE)
    a = torch.cosh(gl)
    b = z0 * torch.sinh(gl)
    c = torch.sinh(gl) / (z0 + 1e-300)
    d = torch.cosh(gl)
    denom = a + b / Z_REF + c * Z_REF + d + 1e-30
    s = torch.zeros((*a.shape, 2, 2), dtype=COMPLEX_DTYPE, device=params.device)
    s[..., 0, 0] = (a + b / Z_REF - c * Z_REF - d) / denom
    s[..., 0, 1] = 2.0 * (a * d - b * c) / denom
    s[..., 1, 0] = 2.0 / denom
    s[..., 1, 1] = (-a + b / Z_REF - c * Z_REF + d) / denom
    return s


def circuit_params_to_s_np(params, length_um, freqs_hz):
    params_t = torch.tensor(np.asarray(params, dtype=np.float64).reshape(1, -1), dtype=REAL_DTYPE)
    length_t = torch.tensor([float(length_um)], dtype=REAL_DTYPE)
    with torch.no_grad():
        return circuit_params_to_s_torch(params_t, length_t, freqs_hz).cpu().numpy()[0]


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def predict_params(model, x_raw, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(x_raw), 256):
            xb = torch.tensor(x_raw[start : start + 256], dtype=REAL_DTYPE, device=device)
            rows.append(torch.clamp(model(xb), min=1e-30).cpu().numpy())
    return np.vstack(rows)


def evaluate_model(model, df, x_raw, length_um, s_target, freq, device, label):
    pred_params = predict_params(model, x_raw, device)
    metrics = []
    pred_df = df[["source_root", "source_split", "split", "idx", "snp_path"] + TARGET_PARAMS].copy()
    for i, row in df.iterrows():
        pred_s = circuit_params_to_s_np(pred_params[i], length_um[i], freq)
        metrics.append(
            {
                "model": label,
                "source_root": row["source_root"],
                "source_split": row["source_split"],
                "split": row["split"],
                "idx": int(row["idx"]),
                "file": Path(row["snp_path"]).name,
                "sparam_mse": float(np.mean(np.abs(pred_s - s_target[i]) ** 2)),
                "s11_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 0]) - db20(s_target[i, :, 0, 0])))),
                "s21_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(s_target[i, :, 1, 0])))),
                "s12_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 1]) - db20(s_target[i, :, 0, 1])))),
                "s22_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 1]) - db20(s_target[i, :, 1, 1])))),
            }
        )
    for j, name in enumerate(TARGET_PARAMS):
        pred_df[f"pred_{name}"] = pred_params[:, j]
    return pd.DataFrame(metrics), pred_df


def train_sparam_model(model, x_raw, length_um, s_target, freq, masks, device):
    x_train = torch.tensor(x_raw[masks["train"]], dtype=REAL_DTYPE)
    l_train = torch.tensor(length_um[masks["train"]], dtype=REAL_DTYPE)
    s_train = torch.tensor(s_target[masks["train"]], dtype=COMPLEX_DTYPE)
    loader = DataLoader(TensorDataset(x_train, l_train, s_train), batch_size=SPARAM_BATCH_SIZE, shuffle=True)

    x_val = torch.tensor(x_raw[masks["val"]], dtype=REAL_DTYPE, device=device)
    l_val = torch.tensor(length_um[masks["val"]], dtype=REAL_DTYPE, device=device)
    s_val = torch.tensor(s_target[masks["val"]], dtype=COMPLEX_DTYPE, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=SPARAM_LR, weight_decay=SPARAM_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=45, factor=0.5)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    rows = []

    for epoch in range(1, SPARAM_EPOCHS + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, lb, sb in loader:
            xb = xb.to(device)
            lb = lb.to(device)
            sb = sb.to(device)
            optimizer.zero_grad(set_to_none=True)
            params = torch.clamp(model(xb), min=1e-30)
            pred_s = circuit_params_to_s_torch(params, lb, freq)
            loss = torch.mean(torch.abs(pred_s - sb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            n_seen += len(xb)

        train_loss = total / max(n_seen, 1)
        model.eval()
        with torch.no_grad():
            val_params = torch.clamp(model(x_val), min=1e-30)
            val_s = circuit_params_to_s_torch(val_params, l_val, freq)
            val_loss = torch.mean(torch.abs(val_s - s_val) ** 2).item()
        scheduler.step(val_loss)
        rows.append({"epoch": epoch, "train_sparam_loss": train_loss, "val_sparam_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"  [sparam] epoch={epoch}, train={train_loss:.6e}, val={val_loss:.6e}", flush=True)
        if stale >= SPARAM_PATIENCE:
            print(f"  [sparam] early stop at epoch={epoch}, best_val={best_val:.6e}", flush=True)
            break

    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def save_compare_plots(dataset_name, device_name, out_dir, df, matlab_metrics, matlab_params, ft_params, s_target, freq):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    worst = matlab_metrics[matlab_metrics["split"] == "test"].sort_values("sparam_mse", ascending=False).head(PLOT_WORST_N)
    for _, metric in worst.iterrows():
        matches = df.index[(df["split"] == metric["split"]) & (df["idx"] == metric["idx"])]
        if len(matches) == 0:
            continue
        i = int(matches[0])
        length_col = DEVICE_CONFIGS[device_name]["length_column"]
        mat_p = matlab_params.loc[i, [f"pred_{name}" for name in TARGET_PARAMS]].to_numpy(dtype=np.float64)
        ft_p = ft_params.loc[i, [f"pred_{name}" for name in TARGET_PARAMS]].to_numpy(dtype=np.float64)
        mat_s = circuit_params_to_s_np(mat_p, df.loc[i, length_col], freq)
        ft_s = circuit_params_to_s_np(ft_p, df.loc[i, length_col], freq)
        save_s_compare_plot(
            plot_dir / f"{dataset_name}_{device_name.lower()}_test_dut{int(metric['idx'])}.png",
            freq,
            s_target[i],
            mat_s,
            ft_s,
            f"{dataset_name} {device_name} test dut{int(metric['idx'])}",
        )


def save_s_compare_plot(path, freq, target_s, matlab_s, finetuned_s, title):
    freq_ghz = freq / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=140)
    fig.suptitle(title, x=0.02, y=0.985, ha="left")
    for ax, (m, n, label) in zip(axes.ravel(), [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]):
        ax.plot(freq_ghz, db20(target_s[:, m, n]), label="HFSS", color="black", linewidth=1.8)
        ax.plot(freq_ghz, db20(matlab_s[:, m, n]), label="MATLAB param NN", color="#2563eb", linestyle="--", linewidth=1.4)
        ax.plot(freq_ghz, db20(finetuned_s[:, m, n]), label="S-param finetuned", color="#dc2626", linestyle="-.", linewidth=1.4)
        ax.set_title(label)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path)
    plt.close(fig)


def train_one(dataset_name, device_name, device):
    print(f"\nFine-tuning {dataset_name} / {device_name}", flush=True)
    config = DEVICE_CONFIGS[device_name]
    out_dir = OUTPUT_ROOT / dataset_name / device_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_param_table(dataset_name, device_name)
    df = df.dropna(subset=config["features"] + TARGET_PARAMS + [config["length_column"]]).reset_index(drop=True)
    masks = split_masks(df)
    x_raw = df[config["features"]].to_numpy(dtype=np.float64)
    length_um = df[config["length_column"]].to_numpy(dtype=np.float64)
    s_target, freq = load_s_targets(df)
    print(
        f"  samples: train={int(masks['train'].sum())}, val={int(masks['val'].sum())}, test={int(masks['test'].sum())}",
        flush=True,
    )

    model = MatlabMultiParamNet(dataset_name, device_name).to(dtype=REAL_DTYPE, device=device)
    matlab_metrics, matlab_params = evaluate_model(model, df, x_raw, length_um, s_target, freq, device, "matlab_param_nn")
    history = train_sparam_model(model, x_raw, length_um, s_target, freq, masks, device)
    finetuned_metrics, finetuned_params = evaluate_model(model, df, x_raw, length_um, s_target, freq, device, "sparam_finetuned")

    combined_metrics = pd.concat([matlab_metrics, finetuned_metrics], ignore_index=True)
    combined_metrics.to_csv(out_dir / "metrics_before_after.csv", index=False, encoding="utf-8-sig")
    matlab_params.to_csv(out_dir / "matlab_predicted_circuit_params.csv", index=False, encoding="utf-8-sig")
    finetuned_params.to_csv(out_dir / "finetuned_predicted_circuit_params.csv", index=False, encoding="utf-8-sig")
    history.to_csv(out_dir / "sparam_finetune_history.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "dataset": dataset_name,
                "device_name": device_name,
                "feature_columns": config["features"],
                "target_params": TARGET_PARAMS,
                "length_column": config["length_column"],
                "freq_hz": freq.tolist(),
                "initial_model_dir": str(MATLAB_MODEL_ROOT / dataset_name),
                "model_type": "matlab_exported_rdl_param_nn_sparam_finetuned",
                "sparam_lr": SPARAM_LR,
                "sparam_epochs": SPARAM_EPOCHS,
            },
        },
        out_dir / "matlab_param_net_sparam_finetuned.pt",
    )
    save_compare_plots(dataset_name, device_name, out_dir, df, matlab_metrics, matlab_params, finetuned_params, s_target, freq)

    summary = combined_metrics.groupby(["model", "split"])[["sparam_mse", "s11_db_mae", "s21_db_mae", "s12_db_mae", "s22_db_mae"]].mean()
    print(summary.to_string(), flush=True)
    return combined_metrics


def main():
    set_seed(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    summary_rows = []
    report = {}
    for dataset_name in DATASET_NAMES:
        report[dataset_name] = {}
        for device_name in DEVICE_NAMES:
            metrics = train_one(dataset_name, device_name, device)
            summary = (
                metrics.groupby(["model", "split"])[["sparam_mse", "s11_db_mae", "s21_db_mae", "s12_db_mae", "s22_db_mae"]]
                .mean()
                .reset_index()
            )
            summary.insert(0, "device", device_name)
            summary.insert(0, "dataset", dataset_name)
            summary_rows.append(summary)
            report[dataset_name][device_name] = summary.drop(columns=["dataset", "device"]).to_dict(orient="records")

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_file = OUTPUT_ROOT / "summary_metrics.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    with open(OUTPUT_ROOT / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "workflow": "five_lhs_rdl_matlab_param_nn_then_sparam_finetune",
                "param_table_root": str(PARAM_TABLE_ROOT),
                "matlab_model_root": str(MATLAB_MODEL_ROOT),
                "output_root": str(OUTPUT_ROOT),
                "datasets": DATASET_NAMES,
                "devices": DEVICE_NAMES,
                "metrics": report,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nDone. Summary: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
