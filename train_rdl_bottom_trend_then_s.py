import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import savemat
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = os.path.dirname(__file__)
PARAM_NAMES = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
GEOM_NAMES = ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl"]
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


def robust_zscore(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 0.6745 * (values - median) / max(mad, 1e-12)


def select_trend_samples(df, score_threshold=3.5, rmse_quantile=0.95):
    x = df[GEOM_NAMES].to_numpy(dtype=float)
    y = np.log(np.maximum(df[PARAM_NAMES].to_numpy(dtype=float), 1e-30))

    base_mask = df["rmse"].to_numpy() <= df["rmse"].quantile(rmse_quantile)
    pred = np.zeros_like(y)
    residual = np.zeros_like(y)

    for j in range(y.shape[1]):
        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000),
        )
        model.fit(x[base_mask], y[base_mask, j])
        pred[:, j] = model.predict(x)
        residual[:, j] = y[:, j] - pred[:, j]

    z = np.zeros_like(residual)
    for j in range(residual.shape[1]):
        z[:, j] = robust_zscore(residual[:, j])

    score = np.max(np.abs(z), axis=1)
    mask = base_mask & (score <= score_threshold)

    selected = df.copy()
    selected["trend_score"] = score
    selected["trend_selected"] = mask
    return selected, mask, residual, z


def parse_s2p_header_vars(path):
    variables = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                break
            if line.startswith("!") and "=" in line:
                name, rest = line[1:].split("=", 1)
                match = re.search(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", rest)
                if match:
                    variables[name.strip()] = float(match.group(1))
    return variables


def attach_dut_indices(df, snp_dir):
    if "dut_index" in df.columns:
        return df

    key_to_dut = {}
    for filename in os.listdir(snp_dir):
        if not filename.lower().endswith((".s2p", ".s1p", ".s3p", ".s4p")):
            continue
        match = re.search(r"dut(\d+)", filename, re.IGNORECASE)
        if not match:
            continue
        dut_idx = int(match.group(1))
        variables = parse_s2p_header_vars(os.path.join(snp_dir, filename))
        try:
            key = (
                round(variables["ldown"], 6),
                round(variables["wdown"], 6),
                round(variables["tdown"], 6),
                round(variables["htsv"], 6),
                round(variables["p1"], 6),
            )
        except KeyError:
            continue
        key_to_dut[key] = dut_idx

    dut_indices = []
    missing_rows = []
    for row_idx, row in df.iterrows():
        key = tuple(round(float(row[name]), 6) for name in GEOM_NAMES)
        dut_idx = key_to_dut.get(key)
        if dut_idx is None:
            missing_rows.append(int(row_idx))
            dut_idx = -1
        dut_indices.append(dut_idx)

    df = df.copy()
    df["dut_index"] = dut_indices
    if missing_rows:
        print(f"[warn] {len(missing_rows)} CSV rows could not be matched to dut*.s2p by geometry.")
    else:
        print(f">>> matched all {len(df)} CSV rows to dut*.s2p by geometry.")
    return df


def load_s_dataset(df, snp_dir, max_points=300):
    s_list = []
    freq_ref = None
    valid = []
    for idx in range(len(df)):
        dut_idx = int(df.iloc[idx]["dut_index"]) if "dut_index" in df.columns else idx
        if dut_idx < 0:
            valid.append(False)
            continue
        path = os.path.join(snp_dir, f"dut{dut_idx}.s2p")
        if not os.path.exists(path):
            valid.append(False)
            continue
        nw = rf.Network(path)
        s = nw.s[:max_points]
        freq = nw.f[:max_points]
        if freq_ref is None:
            freq_ref = freq
        if len(freq) != len(freq_ref) or not np.allclose(freq, freq_ref):
            valid.append(False)
            continue
        s_list.append(s)
        valid.append(True)

    valid = np.asarray(valid, dtype=bool)
    s_arr = np.asarray(s_list, dtype=np.complex128)
    return freq_ref, s_arr, valid


def params_to_s_torch(params, length_um, freq_hz):
    R1, R2, R3 = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    L1, L2, L3 = params[:, 3:4] * 1e-9, params[:, 4:5] * 1e-9, params[:, 5:6] * 1e-9
    Cox, Csi, Rsi = params[:, 6:7] * 1e-12, params[:, 7:8] * 1e-12, params[:, 8:9]

    omega = 2 * torch.pi * freq_hz.reshape(1, -1)
    length_m = length_um.reshape(-1, 1) * 1e-6

    R_rlgc = (
        (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2)
        / ((R1 + R2) ** 2 + omega**2 * L2**2)
        + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    )
    L_rlgc = (
        (R1**2 * L2) / ((R1 + R2) ** 2 + omega**2 * L2**2)
        + L3 * R3**2 / (R3**2 + omega**2 * L3**2)
        + L1
    )
    G_rlgc = omega**2 * Rsi * Cox**2 / (1 + omega**2 * Rsi**2 * (Cox + Csi) ** 2)
    C_rlgc = (
        Cox
        + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)
        / (1 + omega**2 * Rsi**2 * (Cox + Csi) ** 2)
    )

    z_series = torch.complex(R_rlgc, omega * L_rlgc)
    y_shunt = torch.complex(G_rlgc, omega * C_rlgc)
    z0 = torch.sqrt(z_series / y_shunt)
    gamma = torch.sqrt(z_series * y_shunt)
    gl = gamma * length_m
    A = torch.cosh(gl)
    B = z0 * torch.sinh(gl)
    C = torch.sinh(gl) / z0
    D = torch.cosh(gl)
    denom = A + B / Z_REF + C * Z_REF + D

    S11 = (A + B / Z_REF - C * Z_REF - D) / denom
    S12 = 2 * (A * D - B * C) / denom
    S21 = 2 / denom
    S22 = (-A + B / Z_REF - C * Z_REF + D) / denom

    return torch.stack(
        [
            torch.stack([S11, S12], dim=-1),
            torch.stack([S21, S22], dim=-1),
        ],
        dim=-2,
    )


def s_loss(pred_s, target_s):
    diff = pred_s - target_s
    return torch.mean(torch.real(diff * torch.conj(diff)))


def s_db_loss(pred_s, target_s):
    pred_db = 20.0 * torch.log10(torch.clamp(torch.abs(pred_s), min=1e-12))
    target_db = 20.0 * torch.log10(torch.clamp(torch.abs(target_s), min=1e-12))
    return torch.mean((pred_db - target_db) ** 2)


def save_matlab_like_exports(model, stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "stats": stats,
        "param_names": PARAM_NAMES,
        "geom_names": GEOM_NAMES,
    }
    safe_torch_save(state, os.path.join(output_dir, "rdl_bottom_param_net.pt"))

    savemat(
        os.path.join(output_dir, "rdl_bottom_param_net_stats.mat"),
        {
            "x_mean": stats["x_mean"],
            "x_std": stats["x_std"],
            "y_mean": stats["y_mean"],
            "y_std": stats["y_std"],
            "param_names": np.asarray(PARAM_NAMES, dtype=object),
            "geom_names": np.asarray(GEOM_NAMES, dtype=object),
        },
    )


def safe_torch_save(obj, path):
    if os.path.exists(path):
        os.remove(path)
    torch.save(obj, path)


def predict_params(model, x_norm, stats, device):
    model.eval()
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm, dtype=torch.float64, device=device))
        y_log = y_norm.cpu().numpy() * stats["y_std"] + stats["y_mean"]
    return np.exp(y_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "RDL_Bottom_TD_2.csv"))
    parser.add_argument("--snp-dir", default=os.path.join(BASE_DIR, "RDL_Bottom_Snp"))
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "RDL_Bottom_trend_sparam_training"))
    parser.add_argument("--score-threshold", type=float, default=3.5)
    parser.add_argument("--supervised-epochs", type=int, default=2000)
    parser.add_argument("--fine-epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--s-batch-size", type=int, default=24)
    parser.add_argument("--fine-reg-weight", type=float, default=2e-4)
    parser.add_argument("--fine-db-weight", type=float, default=1e-4)
    parser.add_argument("--max-points", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = attach_dut_indices(df, args.snp_dir)
    selected_df, mask, residual, z = select_trend_samples(df, args.score_threshold)
    selected_path = os.path.join(args.out_dir, "RDL_Bottom_TD_trend_selected.csv")
    selected_df.to_csv(selected_path, index=False)
    selected_df[selected_df["trend_selected"]].to_csv(
        os.path.join(args.out_dir, "RDL_Bottom_TD_trend_selected_only.csv"),
        index=False,
    )

    freq, s_np, valid_s = load_s_dataset(df, args.snp_dir, max_points=args.max_points)
    train_mask = mask & valid_s

    x = df[GEOM_NAMES].to_numpy(dtype=np.float64)
    y_log = np.log(np.maximum(df[PARAM_NAMES].to_numpy(dtype=np.float64), 1e-30))
    x_mean, x_std = x[train_mask].mean(axis=0), np.maximum(x[train_mask].std(axis=0), 1e-12)
    y_mean, y_std = y_log[train_mask].mean(axis=0), np.maximum(y_log[train_mask].std(axis=0), 1e-12)
    x_norm = (x - x_mean) / x_std
    y_norm = (y_log - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    model = ParamNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=60, factor=0.5)

    ds = TensorDataset(
        torch.tensor(x_norm[train_mask], dtype=torch.float64),
        torch.tensor(y_norm[train_mask], dtype=torch.float64),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    history = {"epoch": [], "supervised_loss": [], "fine_loss": [], "fine_s_loss": [], "fine_reg_loss": []}

    best_sup = float("inf")
    best_sup_state = None
    for epoch in range(1, args.supervised_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        loss_epoch = total / max(count, 1)
        scheduler.step(loss_epoch)
        if loss_epoch < best_sup:
            best_sup = loss_epoch
            best_sup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0:
            print(f"[supervised] epoch={epoch}, loss={loss_epoch:.6e}, selected={int(train_mask.sum())}")
        history["epoch"].append(epoch)
        history["supervised_loss"].append(loss_epoch)
        history["fine_loss"].append(np.nan)
        history["fine_s_loss"].append(np.nan)
        history["fine_reg_loss"].append(np.nan)
    if best_sup_state is not None:
        model.load_state_dict(best_sup_state)
        safe_torch_save(best_sup_state, os.path.join(args.out_dir, "param_net_supervised.pth"))

    s_indices = np.where(valid_s)[0]
    s_target = torch.tensor(s_np, dtype=torch.complex128, device=device)
    freq_t = torch.tensor(freq, dtype=torch.float64, device=device)
    x_all_t = torch.tensor(x_norm[s_indices], dtype=torch.float64, device=device)
    length_t = torch.tensor(df.loc[s_indices, "l_rdl"].to_numpy(dtype=np.float64), dtype=torch.float64, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=torch.float64, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad():
        y_anchor_t = model(x_all_t).detach()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-7)
    best_fine = float("inf")
    best_fine_state = None
    n_s = len(s_indices)
    for epoch in range(1, args.fine_epochs + 1):
        perm = torch.randperm(n_s, device=device)
        total = total_s = total_reg = 0.0
        count = 0
        model.train()
        for start in range(0, n_s, args.s_batch_size):
            batch_idx = perm[start : start + args.s_batch_size]
            xb = x_all_t[batch_idx]
            y_ref = y_anchor_t[batch_idx]
            length_b = length_t[batch_idx]
            target_b = s_target[batch_idx]

            y_pred_norm = model(xb)
            params = torch.exp(y_pred_norm * y_std_t + y_mean_t)
            pred_s = params_to_s_torch(params, length_b, freq_t)

            loss_s = s_loss(pred_s, target_b)
            loss_db = s_db_loss(pred_s, target_b)
            loss_reg = F.mse_loss(y_pred_norm, y_ref)
            loss = loss_s + args.fine_db_weight * loss_db + args.fine_reg_weight * loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            bs = len(batch_idx)
            total += float(loss.detach().cpu()) * bs
            total_s += float(loss_s.detach().cpu()) * bs
            total_reg += float(loss_reg.detach().cpu()) * bs
            count += bs

        loss_epoch = total / max(count, 1)
        loss_s_epoch = total_s / max(count, 1)
        loss_reg_epoch = total_reg / max(count, 1)
        if loss_epoch < best_fine:
            best_fine = loss_epoch
            best_fine_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0:
            print(
                f"[fine] epoch={epoch}, loss={loss_epoch:.6e}, "
                f"loss_s={loss_s_epoch:.6e}, loss_reg={loss_reg_epoch:.6e}"
            )
        history["epoch"].append(args.supervised_epochs + epoch)
        history["supervised_loss"].append(np.nan)
        history["fine_loss"].append(loss_epoch)
        history["fine_s_loss"].append(loss_s_epoch)
        history["fine_reg_loss"].append(loss_reg_epoch)

    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    if best_fine_state is not None:
        model.load_state_dict(best_fine_state)
        safe_torch_save(best_fine_state, os.path.join(args.out_dir, "param_net_s_finetuned.pth"))
    save_matlab_like_exports(model, stats, args.out_dir)

    pred_params = predict_params(model, x_norm, stats, device)
    pred_df = df[GEOM_NAMES].copy()
    for idx, name in enumerate(PARAM_NAMES):
        pred_df[name] = pred_params[:, idx]
    pred_df.to_csv(os.path.join(args.out_dir, "RDL_Bottom_TD_nn_s_finetuned_params.csv"), index=False)

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_history.csv"), index=False)
    report = {
        "selected_count": int(train_mask.sum()),
        "total_count": int(len(df)),
        "selected_csv": selected_path,
        "best_supervised_loss": best_sup,
        "best_fine_loss": best_fine,
        "device": str(device),
    }
    with open(os.path.join(args.out_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, name in zip(axes.ravel(), PARAM_NAMES):
        original = df[name].sort_values().reset_index(drop=True)
        predicted = pred_df[name].sort_values().reset_index(drop=True)
        ax.plot(original, label="original", linewidth=1.2)
        ax.plot(predicted, label="NN fine-tuned", linewidth=1.2)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "parameter_distribution_compare.png"), dpi=180)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
