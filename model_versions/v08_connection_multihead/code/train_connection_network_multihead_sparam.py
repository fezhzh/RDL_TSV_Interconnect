# -*- coding: utf-8 -*-
"""Train an 8-head connection-network NN with full-structure S-parameter loss.

Run this file directly in VS Code. The model uses one shared trunk and one head
per connection position. Each head predicts the 7 with-Cn3 connection-network
scale parameters for that position. Device lengths stay fixed at the 0.95 scale
used by ``Calc_SP_and_Opt2.py``.
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
from torch.utils.data import DataLoader, Dataset, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V08_CODE_DIR = PROJECT_ROOT / "model_versions" / "v08_connection_multihead" / "code"
V07_CODE_DIR = PROJECT_ROOT / "model_versions" / "v07_connection_param_and_sparam_finetune" / "code"
V03_CODE_DIR = PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "code"
SPARAM_CODE_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "code"
for path in [V08_CODE_DIR, V07_CODE_DIR, V03_CODE_DIR, SPARAM_CODE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Calc_SP_and_Opt2 as opt2
import fine_tune_connection_network_on_sparams as sfit
import train_connection_network_params as param_train
import train_single_device_sparam_model as single_device


SOURCE_MODEL_DIR = PROJECT_ROOT / "model_versions" / "v08_connection_multihead" / "results" / "connection_network_multihead_sparam_with_cn3_refined_devices_s21_filtered_round4"
CHECKPOINT_PATH = SOURCE_MODEL_DIR / "connection_param_multihead_net.pt"
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v08_connection_multihead" / "results" / "connection_network_multihead_sparam_with_cn3_rigorous_unconstrained"

# Configure these values before running directly from VS Code.
RANDOM_SEED = 20260629
BATCH_SIZE_PARAM = 128
BATCH_SIZE_SPARAM = 16
PARAM_PRETRAIN_EPOCHS = 1200
SPARAM_EPOCHS = 600
PARAM_LR = 8e-4
SPARAM_LR = 2e-5
WEIGHT_DECAY = 1e-8
PATIENCE_PARAM = 180
PATIENCE_SPARAM = 120
PRINT_EVERY = 20
USE_CUDA_IF_AVAILABLE = True
PARAM_ANCHOR_WEIGHT = 1e-6
LOAD_CHECKPOINT = False
RUN_PARAM_PRETRAIN = True
USE_REFINED_SINGLE_DEVICE_MODELS = True
EXCLUDE_BAD_HFSS_S21 = True
BAD_HFSS_S21_MEAN_DB_THRESHOLD = -15.0
STRICT_TEST_SPLIT = True
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
HARD_SAMPLE_WEIGHTING = False
HARD_SAMPLE_TRAIN_ON_ALL_FILTERED = False
HARD_SAMPLE_WEIGHT_FACTOR = 8.0
HARD_SAMPLE_WEIGHT_POWER = 1.0
HARD_SAMPLE_METRIC_COLUMN = "multihead_mse_vs_hfss"
PLOT_SPLIT = "test"
PLOT_DUT_LIMIT = 10
PLOT_SORT_COLUMN = "multihead_mse_vs_hfss"
PLOT_SORT_ASCENDING = False
PHYSICAL_POSITIVE_OUTPUT = False
PHYSICAL_LOWER_EPS = 1e-9
PHYSICAL_UPPER_SCALE = 1e5

REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128

SINGLE_DEVICE_MODEL_DIRS = {
    "RDL_Top": PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "results" / "single_device_sparam_RDL_Top_mat4_init_sparam_noanchor",
    "RDL_Bottom": PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "results" / "single_device_sparam_RDL_Bottom_mat4_init_sparam_noanchor",
    "TSV": PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "results" / "single_device_sparam_TSV_mat4_init_sparam_noanchor",
}


class MultiHeadConnectionNet(nn.Module):
    def __init__(self, input_dim, connection_count=8, head_dim=7):
        super().__init__()
        self.connection_count = connection_count
        self.head_dim = head_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.LayerNorm(64),
                    nn.Linear(64, head_dim),
                )
                for _ in range(connection_count)
            ]
        )

    def forward(self, x):
        z = self.trunk(x)
        outputs = [head(z) for head in self.heads]
        return torch.cat(outputs, dim=1)


class SParamDataset(Dataset):
    def __init__(self, indices, x_norm, y_norm, base_abcds, target_s, dut_indices, sample_weights=None):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.base_abcds = base_abcds
        self.target_s = target_s
        self.dut_indices = dut_indices
        if sample_weights is None:
            self.sample_weights = np.ones(len(dut_indices), dtype=np.float64)
        else:
            self.sample_weights = np.asarray(sample_weights, dtype=np.float64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        return (
            self.dut_indices[idx],
            self.x_norm[idx],
            self.y_norm[idx],
            self.base_abcds[idx],
            self.target_s[idx],
            self.sample_weights[idx],
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_refined_single_device_models():
    models = {}
    for device_name, model_dir in SINGLE_DEVICE_MODEL_DIRS.items():
        checkpoint_path = model_dir / "single_device_sparam_net.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing refined single-device model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        config = single_device.DEVICE_CONFIGS[device_name]
        model = single_device.Mat4InitializedDeviceNet(config, metadata).to(dtype=REAL_DTYPE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[device_name] = {"model": model, "metadata": metadata}
    return models


def predict_refined_device_params(model_info, features):
    metadata = model_info["metadata"]
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    x_norm = (x - np.asarray(metadata["x_mean"], dtype=np.float64)) / np.asarray(metadata["x_std"], dtype=np.float64)
    with torch.no_grad():
        pred_norm = model_info["model"](torch.tensor(x_norm, dtype=REAL_DTYPE))
    y_mean = np.asarray(metadata["y_log_mean"], dtype=np.float64).reshape(1, -1)
    y_std = np.asarray(metadata["y_log_std"], dtype=np.float64).reshape(1, -1)
    params = np.exp(np.clip(pred_norm.cpu().numpy() * y_std + y_mean, -40.0, 40.0))[0]
    return {name: float(value) for name, value in zip(single_device.TARGET_PARAMS, params)}


def build_refined_base_abcds(params, freqs, refined_models):
    features_top = np.array([params["lrdl"], params["wrdl"], params["trdl"], params["htsv"], params["p1"]], dtype=np.float64)
    features_bot = np.array([params["ldown"], params["wdown"], params["tdown"], params["htsv"], params["p1"]], dtype=np.float64)
    features_tsv = np.array([params["dtsv"], params["htsv"], params["p1"]], dtype=np.float64)

    cp_top = predict_refined_device_params(refined_models["RDL_Top"], features_top)
    cp_bot = predict_refined_device_params(refined_models["RDL_Bottom"], features_bot)
    cp_tsv = predict_refined_device_params(refined_models["TSV"], features_tsv)

    abcd_top = opt2.s2abcd(opt2.calculate_S_parameters(cp_top, params["lrdl"] * opt2.DEVICE_LENGTH_SCALE, freqs))
    abcd_bot = opt2.s2abcd(opt2.calculate_S_parameters(cp_bot, params["ldown"] * opt2.DEVICE_LENGTH_SCALE, freqs))
    abcd_tsv = opt2.s2abcd(opt2.calculate_S_parameters(cp_tsv, params["htsv"] * opt2.DEVICE_LENGTH_SCALE, freqs))
    return [abcd_top, abcd_tsv, abcd_bot, abcd_tsv, abcd_top, abcd_tsv, abcd_bot, abcd_tsv, abcd_top]


def s21_mean_db(s_network):
    s21 = np.abs(s_network[:, 1, 0])
    return 20.0 * np.log10(max(float(np.mean(s21)), 1e-30))


def filter_bad_hfss_s21(dut_df):
    if not EXCLUDE_BAD_HFSS_S21:
        return dut_df.reset_index(drop=True), pd.DataFrame()

    rows = []
    for dut_idx in dut_df["dut_index"].astype(int):
        hfss_path = opt2.S2P_DIR / f"dut{dut_idx}.s2p"
        hfss_nw = rf.Network(str(hfss_path))
        rows.append(
            {
                "dut_index": dut_idx,
                "file": hfss_path.name,
                "s21_mean_db": s21_mean_db(hfss_nw.s),
                "s21_max_db": 20.0 * np.log10(max(float(np.max(np.abs(hfss_nw.s[:, 1, 0]))), 1e-30)),
                "s21_min_db": 20.0 * np.log10(max(float(np.min(np.abs(hfss_nw.s[:, 1, 0]))), 1e-30)),
            }
        )
    s21_df = pd.DataFrame(rows)
    bad_df = s21_df[s21_df["s21_mean_db"] < BAD_HFSS_S21_MEAN_DB_THRESHOLD].copy()
    if bad_df.empty:
        print("S21 filter: no bad HFSS samples excluded", flush=True)
        return dut_df.reset_index(drop=True), bad_df

    keep_ids = set(s21_df.loc[s21_df["s21_mean_db"] >= BAD_HFSS_S21_MEAN_DB_THRESHOLD, "dut_index"].astype(int))
    filtered_df = dut_df[dut_df["dut_index"].astype(int).isin(keep_ids)].reset_index(drop=True)
    print(
        f"S21 filter: excluded {len(bad_df)} DUTs with mean S21 < {BAD_HFSS_S21_MEAN_DB_THRESHOLD:.1f} dB",
        flush=True,
    )
    print(bad_df[["dut_index", "s21_mean_db", "s21_max_db", "s21_min_db"]].to_string(index=False), flush=True)
    return filtered_df, bad_df.reset_index(drop=True)


def build_hard_sample_weights(dut_indices):
    weights = np.ones(len(dut_indices), dtype=np.float64)
    if not HARD_SAMPLE_WEIGHTING:
        return weights

    metrics_path = SOURCE_MODEL_DIR / "multihead_sparam_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing hard-sample metrics: {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)
    if HARD_SAMPLE_METRIC_COLUMN not in metrics_df.columns:
        raise ValueError(f"Missing metric column {HARD_SAMPLE_METRIC_COLUMN!r} in {metrics_path}")

    metric_map = {
        int(row["dut_index"]): float(row[HARD_SAMPLE_METRIC_COLUMN])
        for _, row in metrics_df.dropna(subset=[HARD_SAMPLE_METRIC_COLUMN]).iterrows()
    }
    values = np.asarray([metric_map.get(int(dut_idx), 0.0) for dut_idx in dut_indices], dtype=np.float64)
    max_value = max(float(values.max()), 1e-30)
    weights = 1.0 + HARD_SAMPLE_WEIGHT_FACTOR * np.power(values / max_value, HARD_SAMPLE_WEIGHT_POWER)
    print(
        f"Hard-sample weighting: min={weights.min():.3f}, median={np.median(weights):.3f}, max={weights.max():.3f}",
        flush=True,
    )
    return weights


def split_train_val_test(dut_df):
    dut_ids = dut_df["dut_index"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = dut_ids.copy()
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, min(n_total - 2, int(round(n_total * TRAIN_RATIO))))
    n_val = max(1, min(n_total - n_train - 1, int(round(n_total * VAL_RATIO))))
    train_ids = set(shuffled[:n_train].tolist())
    val_ids = set(shuffled[n_train : n_train + n_val].tolist())
    test_ids = set(shuffled[n_train + n_val :].tolist())
    train_mask = dut_df["dut_index"].isin(train_ids).to_numpy()
    val_mask = dut_df["dut_index"].isin(val_ids).to_numpy()
    test_mask = dut_df["dut_index"].isin(test_ids).to_numpy()
    return train_mask, val_mask, test_mask


def normalize_train(values, train_mask):
    mean = values[train_mask].mean(axis=0)
    std = np.maximum(values[train_mask].std(axis=0), 1e-12)
    return (values - mean) / std, mean, std


def physical_bounds(output_dim):
    lower = np.full(output_dim, PHYSICAL_LOWER_EPS, dtype=np.float64)
    upper = np.full(output_dim, PHYSICAL_UPPER_SCALE, dtype=np.float64)
    return lower, upper


def logit_from_physical_targets(y_raw):
    lower, upper = physical_bounds(y_raw.shape[1])
    ratio = (y_raw - lower) / (upper - lower)
    ratio = np.clip(ratio, 1e-8, 1.0 - 1e-8)
    return np.log(ratio / (1.0 - ratio)), lower, upper


def build_matrices_with_test_split(dut_df):
    x_raw = dut_df[param_train.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = dut_df[param_train.TARGET_COLUMNS].to_numpy(dtype=np.float64)
    train_mask, val_mask, test_mask = split_train_val_test(dut_df)

    x_norm, x_mean, x_std = normalize_train(x_raw, train_mask)
    if PHYSICAL_POSITIVE_OUTPUT:
        y_for_model, y_lower, y_upper = logit_from_physical_targets(y_raw)
    else:
        y_for_model = y_raw
        y_lower, y_upper = physical_bounds(y_raw.shape[1])
    y_norm, y_mean, y_std = normalize_train(y_for_model, train_mask)

    metadata = {
        "feature_columns": param_train.STRUCTURE_COLUMNS,
        "target_columns": param_train.TARGET_COLUMNS,
        "scale_columns": param_train.SCALE_COLUMNS,
        "connection_count": param_train.CONNECTION_COUNT,
        "target_variant": param_train.TARGET_VARIANT,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "physical_positive_output": PHYSICAL_POSITIVE_OUTPUT,
        "physical_lower": y_lower.tolist(),
        "physical_upper": y_upper.tolist(),
        "strict_test_split": STRICT_TEST_SPLIT,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
    }
    return x_norm, y_norm, y_raw, train_mask, val_mask, test_mask, metadata


def build_training_arrays():
    connection_df = param_train.load_connection_dataframe()
    dut_df = param_train.build_dut_dataframe(connection_df)
    dut_df, bad_s21_df = filter_bad_hfss_s21(dut_df)
    x_norm, y_norm, y_raw, train_mask, val_mask, test_mask, metadata = build_matrices_with_test_split(dut_df)
    refined_models = load_refined_single_device_models() if USE_REFINED_SINGLE_DEVICE_MODELS else None

    order = np.argsort(dut_df["dut_index"].to_numpy(dtype=np.int64))
    dut_df = dut_df.iloc[order].reset_index(drop=True)
    x_norm = x_norm[order]
    y_norm = y_norm[order]
    y_raw = y_raw[order]
    train_mask = train_mask[order]
    val_mask = val_mask[order]

    base_rows = []
    target_rows = []
    dut_indices = dut_df["dut_index"].to_numpy(dtype=np.int64)
    for n_done, dut_idx in enumerate(dut_indices, start=1):
        hfss_path = opt2.S2P_DIR / f"dut{int(dut_idx)}.s2p"
        hfss_nw, params = sfit.load_hfss_network_with_retry(hfss_path)
        if refined_models is None:
            base_rows.append(np.stack(opt2.build_base_abcds(params, hfss_nw.f), axis=0))
        else:
            base_rows.append(np.stack(build_refined_base_abcds(params, hfss_nw.f, refined_models), axis=0))
        target_rows.append(hfss_nw.s)
        if n_done == 1 or n_done % 200 == 0:
            print(f"预计算级联数据 {n_done}/{len(dut_df)}", flush=True)

    metadata["model_type"] = "multihead_connection_net"
    metadata["head_count"] = param_train.CONNECTION_COUNT
    metadata["head_dim"] = len(param_train.SCALE_COLUMNS)
    metadata["device_length_scale"] = opt2.DEVICE_LENGTH_SCALE
    metadata["use_refined_single_device_models"] = USE_REFINED_SINGLE_DEVICE_MODELS
    metadata["single_device_model_dirs"] = {key: str(value) for key, value in SINGLE_DEVICE_MODEL_DIRS.items()}
    metadata["exclude_bad_hfss_s21"] = EXCLUDE_BAD_HFSS_S21
    metadata["bad_hfss_s21_mean_db_threshold"] = BAD_HFSS_S21_MEAN_DB_THRESHOLD
    metadata["excluded_bad_hfss_dut_indices"] = bad_s21_df["dut_index"].astype(int).tolist() if not bad_s21_df.empty else []
    metadata["hard_sample_weighting"] = HARD_SAMPLE_WEIGHTING
    metadata["hard_sample_train_on_all_filtered"] = HARD_SAMPLE_TRAIN_ON_ALL_FILTERED
    metadata["hard_sample_weight_factor"] = HARD_SAMPLE_WEIGHT_FACTOR
    metadata["hard_sample_weight_power"] = HARD_SAMPLE_WEIGHT_POWER
    metadata["hard_sample_metric_column"] = HARD_SAMPLE_METRIC_COLUMN
    sample_weights = build_hard_sample_weights(dut_indices)

    return (
        dut_df,
        x_norm,
        y_norm,
        y_raw,
        train_mask,
        val_mask,
        test_mask,
        metadata,
        np.stack(base_rows, axis=0),
        np.stack(target_rows, axis=0),
        dut_indices,
        bad_s21_df,
        sample_weights,
    )


def train_param_pretrain(model, arrays, device):
    _, x_norm, y_norm, _, train_mask, val_mask, _, _, _, _, _, _, _ = arrays
    x_train = torch.tensor(x_norm[train_mask], dtype=REAL_DTYPE)
    y_train = torch.tensor(y_norm[train_mask], dtype=REAL_DTYPE)
    x_val = torch.tensor(x_norm[val_mask], dtype=REAL_DTYPE, device=device)
    y_val = torch.tensor(y_norm[val_mask], dtype=REAL_DTYPE, device=device)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE_PARAM, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=60, factor=0.5)

    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, PARAM_PRETRAIN_EPOCHS + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            n_seen += len(xb)

        train_loss = total / max(n_seen, 1)
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(x_val) - y_val) ** 2).item()
        scheduler.step(val_loss)
        history.append({"stage": "param_pretrain", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[param] epoch={epoch}, train={train_loss:.6e}, val={val_loss:.6e}", flush=True)
        if stale >= PATIENCE_PARAM:
            print(f"[param] 早停: epoch={epoch}, best_val={best_val:.6e}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


def normalized_output_to_params(pred_norm, metadata, device):
    y_mean_t = torch.tensor(metadata["y_mean"], dtype=REAL_DTYPE, device=device)
    y_std_t = torch.tensor(metadata["y_std"], dtype=REAL_DTYPE, device=device)
    raw = pred_norm * y_std_t + y_mean_t
    if metadata.get("physical_positive_output", False):
        lower = torch.tensor(metadata["physical_lower"], dtype=REAL_DTYPE, device=device)
        upper = torch.tensor(metadata["physical_upper"], dtype=REAL_DTYPE, device=device)
        return lower + (upper - lower) * torch.sigmoid(raw)
    return raw


def train_sparam(model, arrays, device):
    _, x_norm, y_norm, _, train_mask, val_mask, _, metadata, base_abcds, target_s, dut_indices, _, sample_weights = arrays
    omega = 2.0 * np.pi * rf.Network(str(opt2.S2P_DIR / f"dut{int(dut_indices[0])}.s2p")).f
    train_indices = np.arange(len(dut_indices)) if HARD_SAMPLE_TRAIN_ON_ALL_FILTERED else np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    train_ds = SParamDataset(train_indices, x_norm, y_norm, base_abcds, target_s, dut_indices, sample_weights)
    val_ds = SParamDataset(val_indices, x_norm, y_norm, base_abcds, target_s, dut_indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_SPARAM, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_SPARAM, shuffle=False)

    omega_t = torch.tensor(omega, dtype=REAL_DTYPE, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=SPARAM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)

    best_state = None
    best_val = float("inf")
    stale = 0
    history = []

    def run_loader(loader, training):
        model.train(training)
        total_loss = 0.0
        total_s = 0.0
        total_anchor = 0.0
        n_seen = 0
        for _, x_b, y_b, base_b, target_b, weight_b in loader:
            x_b = x_b.to(device=device, dtype=REAL_DTYPE)
            y_b = y_b.to(device=device, dtype=REAL_DTYPE)
            base_b = base_b.to(device=device, dtype=COMPLEX_DTYPE)
            target_b = target_b.to(device=device, dtype=COMPLEX_DTYPE)
            weight_b = weight_b.to(device=device, dtype=REAL_DTYPE)
            with torch.set_grad_enabled(training):
                pred_norm = model(x_b)
                p_all = normalized_output_to_params(pred_norm, metadata, device).reshape(
                    -1, param_train.CONNECTION_COUNT, len(param_train.SCALE_COLUMNS)
                )
                pred_s = sfit.cascade_with_corrections_torch(base_b, p_all, omega_t)
                loss_s_each = torch.mean(torch.abs(pred_s - target_b) ** 2, dim=(1, 2, 3))
                loss_anchor_each = torch.mean((pred_norm - y_b) ** 2, dim=1)
                if training and HARD_SAMPLE_WEIGHTING:
                    weight_norm = weight_b / torch.mean(weight_b)
                    loss_s = torch.mean(loss_s_each * weight_norm)
                    loss_anchor = torch.mean(loss_anchor_each * weight_norm)
                else:
                    loss_s = torch.mean(loss_s_each)
                    loss_anchor = torch.mean(loss_anchor_each)
                loss = loss_s + PARAM_ANCHOR_WEIGHT * loss_anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError("多头 S 参数训练出现 NaN/Inf")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()

            batch_n = len(x_b)
            total_loss += float(loss.detach().cpu()) * batch_n
            total_s += float(loss_s.detach().cpu()) * batch_n
            total_anchor += float(loss_anchor.detach().cpu()) * batch_n
            n_seen += batch_n
        return total_loss / n_seen, total_s / n_seen, total_anchor / n_seen

    for epoch in range(1, SPARAM_EPOCHS + 1):
        train_loss, train_s, train_anchor = run_loader(train_loader, training=True)
        with torch.no_grad():
            val_loss, val_s, val_anchor = run_loader(val_loader, training=False)
        scheduler.step(val_loss)
        history.append(
            {
                "stage": "sparam_finetune",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_s_loss": train_s,
                "train_anchor_loss": train_anchor,
                "val_loss": val_loss,
                "val_s_loss": val_s,
                "val_anchor_loss": val_anchor,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(
                f"[sparam] epoch={epoch}, train_s={train_s:.6e}, val_s={val_s:.6e}, "
                f"anchor_val={val_anchor:.6e}, lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
        if stale >= PATIENCE_SPARAM:
            print(f"[sparam] 早停: epoch={epoch}, best_val={best_val:.6e}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


def predict_metrics_and_params(model, arrays, device):
    dut_df, x_norm, y_norm, y_raw, train_mask, val_mask, test_mask, metadata, base_abcds, target_s, dut_indices, _, _ = arrays
    omega = 2.0 * np.pi * rf.Network(str(opt2.S2P_DIR / f"dut{int(dut_indices[0])}.s2p")).f
    omega_t = torch.tensor(omega, dtype=REAL_DTYPE, device=device)

    metric_rows = []
    pred_params = []
    model.eval()
    for start in range(0, len(dut_df), BATCH_SIZE_SPARAM):
        stop = min(start + BATCH_SIZE_SPARAM, len(dut_df))
        x_b = torch.tensor(x_norm[start:stop], dtype=REAL_DTYPE, device=device)
        base_b = torch.tensor(base_abcds[start:stop], dtype=COMPLEX_DTYPE, device=device)
        target_b = torch.tensor(target_s[start:stop], dtype=COMPLEX_DTYPE, device=device)
        with torch.no_grad():
            pred_norm = model(x_b)
            pred_params_flat = normalized_output_to_params(pred_norm, metadata, device)
            p_all = pred_params_flat.reshape(-1, param_train.CONNECTION_COUNT, len(param_train.SCALE_COLUMNS))
            pred_s = sfit.cascade_with_corrections_torch(base_b, p_all, omega_t)
        pred_params.append(pred_params_flat.detach().cpu().numpy())
        pred_s_np = pred_s.detach().cpu().numpy()
        target_np = target_b.detach().cpu().numpy()

        for local_i, dut_idx in enumerate(dut_indices[start:stop]):
            optimized_s = rf.Network(str(opt2.OUTPUT_DIR / param_train.TARGET_VARIANT / f"dut{int(dut_idx)}.s2p")).s
            direct_s = opt2.abcd2s(opt2.cascade_direct(list(base_abcds[start + local_i])))
            metric_rows.append(
                {
                    "dut_index": int(dut_idx),
                    "split": (
                        "train"
                        if train_mask[start + local_i]
                        else "val"
                        if val_mask[start + local_i]
                        else "test"
                        if test_mask[start + local_i]
                        else "unused"
                    ),
                    "direct_mse_vs_hfss": opt2.mse(target_np[local_i], direct_s),
                    "optimized_mse_vs_hfss": opt2.mse(target_np[local_i], optimized_s),
                    "multihead_mse_vs_hfss": opt2.mse(target_np[local_i], pred_s_np[local_i]),
                    "multihead_mse_vs_optimized": opt2.mse(optimized_s, pred_s_np[local_i]),
                }
            )

    pred_df = dut_df[["file", "dut_index", "variant"] + param_train.STRUCTURE_COLUMNS].copy()
    pred_arr = np.vstack(pred_params)
    for i, col in enumerate(param_train.TARGET_COLUMNS):
        pred_df[f"target_{col}"] = y_raw[:, i]
        pred_df[f"pred_{col}"] = pred_arr[:, i]
    return pd.DataFrame(metric_rows), pred_df


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def save_plots(metrics_df, pred_df, arrays):
    _, _, _, _, _, _, _, _, base_abcds, _, dut_indices, _, _ = arrays
    plot_df = metrics_df if PLOT_SPLIT == "all" else metrics_df[metrics_df["split"] == PLOT_SPLIT]
    if PLOT_SORT_COLUMN in plot_df.columns:
        plot_df = plot_df.sort_values(PLOT_SORT_COLUMN, ascending=PLOT_SORT_ASCENDING)
    else:
        plot_df = plot_df.sort_values("dut_index")
    plot_dir = OUTPUT_DIR / "multihead_sparam_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_plotted = 0
    for _, metric_row in plot_df.iterrows():
        if PLOT_DUT_LIMIT is not None and n_plotted >= PLOT_DUT_LIMIT:
            break
        dut_idx = int(metric_row["dut_index"])
        array_idx = int(np.where(dut_indices == dut_idx)[0][0])
        hfss_nw = rf.Network(str(opt2.S2P_DIR / f"dut{dut_idx}.s2p"))
        optimized_nw = rf.Network(str(opt2.OUTPUT_DIR / param_train.TARGET_VARIANT / f"dut{dut_idx}.s2p"))
        direct_s = opt2.abcd2s(opt2.cascade_direct(list(base_abcds[array_idx])))
        p_row = pred_df[pred_df["dut_index"] == dut_idx].iloc[0]
        p_all = []
        for conn_idx in range(1, param_train.CONNECTION_COUNT + 1):
            for name in param_train.SCALE_COLUMNS:
                p_all.append(p_row[f"pred_conn{conn_idx}_{name}"])
        pred_s = opt2.abcd2s(
            opt2.cascade_with_corrections(
                list(base_abcds[array_idx]),
                2.0 * np.pi * hfss_nw.f,
                np.asarray(p_all, dtype=np.float64),
                include_cn3=True,
            )
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        fig.suptitle(f"dut{dut_idx}.s2p multi-head S-parameter NN ({metric_row['split']})", x=0.02, y=0.985, ha="left")
        fig.text(
            0.02,
            0.955,
            (
                f"Direct={metric_row['direct_mse_vs_hfss']:.3e} | "
                f"Optimized={metric_row['optimized_mse_vs_hfss']:.3e} | "
                f"Multi-head={metric_row['multihead_mse_vs_hfss']:.3e}"
            ),
            ha="left",
            va="top",
            fontsize=9,
            color="#475569",
        )
        freq_ghz = hfss_nw.f / 1e9
        for ax, (m, n, label) in zip(axes.ravel(), [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]):
            ax.plot(freq_ghz, db20(hfss_nw.s[:, m, n]), label="HFSS", color="black", linewidth=2.0)
            ax.plot(freq_ghz, db20(direct_s[:, m, n]), label="Direct", color="#64748b", linestyle=":", linewidth=1.6)
            ax.plot(freq_ghz, db20(optimized_nw.s[:, m, n]), label="Optimized", color="#dc2626", linestyle="--", linewidth=1.6)
            ax.plot(freq_ghz, db20(pred_s[:, m, n]), label="Multi-head NN", color="#16a34a", linestyle="-.", linewidth=1.6)
            ax.set_title(f"{label} magnitude")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.9, wspace=0.2, hspace=0.3)
        fig.savefig(plot_dir / f"dut{dut_idx}_{metric_row['split']}_multihead_sparam.png", dpi=150)
        plt.close(fig)
        n_plotted += 1
    return plot_dir, n_plotted


def load_multihead_checkpoint_if_enabled(model, device):
    if not LOAD_CHECKPOINT:
        return
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"未找到 multihead 初始模型: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dtype=REAL_DTYPE, device=device)
    print(f"已加载 multihead 初始模型: {CHECKPOINT_PATH}", flush=True)


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)
    arrays = build_training_arrays()
    dut_df, x_norm, _, _, train_mask, val_mask, test_mask, metadata, _, _, _, bad_s21_df, _ = arrays
    print(f"DUT 样本数: {len(dut_df)}", flush=True)
    print(
        f"训练 DUT 数: {int(train_mask.sum())}, 验证 DUT 数: {int(val_mask.sum())}, 测试 DUT 数: {int(test_mask.sum())}",
        flush=True,
    )
    print(f"输入维度: {x_norm.shape[1]}, 输出维度: {len(param_train.TARGET_COLUMNS)}", flush=True)

    model = MultiHeadConnectionNet(
        input_dim=x_norm.shape[1],
        connection_count=param_train.CONNECTION_COUNT,
        head_dim=len(param_train.SCALE_COLUMNS),
    ).to(dtype=REAL_DTYPE, device=device)
    load_multihead_checkpoint_if_enabled(model, device)

    if RUN_PARAM_PRETRAIN:
        param_history = train_param_pretrain(model, arrays, device)
    else:
        param_history = pd.DataFrame()
        print("跳过参数预训练，直接继续 S 参数微调", flush=True)
    sparam_history = train_sparam(model, arrays, device)
    history_df = pd.concat([param_history, sparam_history], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not bad_s21_df.empty:
        bad_s21_df.to_csv(OUTPUT_DIR / "excluded_bad_hfss_s21_samples.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "model_type": "multihead_connection_net",
            "connection_count": param_train.CONNECTION_COUNT,
            "head_dim": len(param_train.SCALE_COLUMNS),
            "param_anchor_weight": PARAM_ANCHOR_WEIGHT,
            "source_checkpoint": str(CHECKPOINT_PATH) if LOAD_CHECKPOINT else None,
            "run_param_pretrain": RUN_PARAM_PRETRAIN,
            "use_refined_single_device_models": USE_REFINED_SINGLE_DEVICE_MODELS,
            "single_device_model_dirs": {key: str(value) for key, value in SINGLE_DEVICE_MODEL_DIRS.items()},
            "exclude_bad_hfss_s21": EXCLUDE_BAD_HFSS_S21,
            "bad_hfss_s21_mean_db_threshold": BAD_HFSS_S21_MEAN_DB_THRESHOLD,
            "excluded_bad_hfss_dut_indices": bad_s21_df["dut_index"].astype(int).tolist() if not bad_s21_df.empty else [],
            "hard_sample_weighting": HARD_SAMPLE_WEIGHTING,
            "hard_sample_train_on_all_filtered": HARD_SAMPLE_TRAIN_ON_ALL_FILTERED,
            "hard_sample_weight_factor": HARD_SAMPLE_WEIGHT_FACTOR,
            "hard_sample_weight_power": HARD_SAMPLE_WEIGHT_POWER,
            "hard_sample_metric_column": HARD_SAMPLE_METRIC_COLUMN,
        },
        OUTPUT_DIR / "connection_param_multihead_net.pt",
    )
    history_df.to_csv(OUTPUT_DIR / "multihead_training_history.csv", index=False, encoding="utf-8-sig")

    metrics_df, pred_df = predict_metrics_and_params(model, arrays, device)
    metrics_df.to_csv(OUTPUT_DIR / "multihead_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUTPUT_DIR / "multihead_param_predictions.csv", index=False, encoding="utf-8-sig")
    plot_dir, n_plots = save_plots(metrics_df, pred_df, arrays)

    val_metrics = metrics_df[metrics_df["split"] == "val"]
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    report = {
        "output_dir": str(OUTPUT_DIR),
        "target_variant": param_train.TARGET_VARIANT,
        "n_dut": int(len(dut_df)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "device": str(device),
        "model_type": "multihead_connection_net",
        "param_pretrain_epochs": int(param_history["epoch"].iloc[-1]) if len(param_history) else 0,
        "sparam_epochs": int(sparam_history["epoch"].iloc[-1]) if len(sparam_history) else 0,
        "best_val_s_loss": float(sparam_history["val_s_loss"].min()) if len(sparam_history) else None,
        "source_checkpoint": str(CHECKPOINT_PATH) if LOAD_CHECKPOINT else None,
        "run_param_pretrain": RUN_PARAM_PRETRAIN,
        "use_refined_single_device_models": USE_REFINED_SINGLE_DEVICE_MODELS,
        "single_device_model_dirs": {key: str(value) for key, value in SINGLE_DEVICE_MODEL_DIRS.items()},
        "exclude_bad_hfss_s21": EXCLUDE_BAD_HFSS_S21,
        "bad_hfss_s21_mean_db_threshold": BAD_HFSS_S21_MEAN_DB_THRESHOLD,
        "excluded_bad_hfss_dut_indices": bad_s21_df["dut_index"].astype(int).tolist() if not bad_s21_df.empty else [],
        "hard_sample_weighting": HARD_SAMPLE_WEIGHTING,
        "hard_sample_train_on_all_filtered": HARD_SAMPLE_TRAIN_ON_ALL_FILTERED,
        "hard_sample_weight_factor": HARD_SAMPLE_WEIGHT_FACTOR,
        "hard_sample_weight_power": HARD_SAMPLE_WEIGHT_POWER,
        "hard_sample_metric_column": HARD_SAMPLE_METRIC_COLUMN,
        "n_plots": int(n_plots),
        "plot_dir": str(plot_dir),
        "val_mean_multihead_mse_vs_hfss": float(val_metrics["multihead_mse_vs_hfss"].mean()),
        "test_mean_multihead_mse_vs_hfss": float(test_metrics["multihead_mse_vs_hfss"].mean()) if len(test_metrics) else None,
        "test_p95_multihead_mse_vs_hfss": float(test_metrics["multihead_mse_vs_hfss"].quantile(0.95)) if len(test_metrics) else None,
        "test_max_multihead_mse_vs_hfss": float(test_metrics["multihead_mse_vs_hfss"].max()) if len(test_metrics) else None,
    }
    with open(OUTPUT_DIR / "multihead_training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n多头 S 参数训练完成", flush=True)
    print(f"模型文件: {OUTPUT_DIR / 'connection_param_multihead_net.pt'}", flush=True)
    print(f"指标 CSV: {OUTPUT_DIR / 'multihead_sparam_metrics.csv'}", flush=True)
    print(f"对比图目录: {plot_dir}", flush=True)
    print("验证集平均 MSE:", flush=True)
    print(
        val_metrics[
            [
                "direct_mse_vs_hfss",
                "optimized_mse_vs_hfss",
                "multihead_mse_vs_hfss",
                "multihead_mse_vs_optimized",
            ]
        ]
        .mean()
        .to_string(),
        flush=True,
    )
    if len(test_metrics):
        print("测试集平均 MSE:", flush=True)
        print(
            test_metrics[
                [
                    "direct_mse_vs_hfss",
                    "optimized_mse_vs_hfss",
                    "multihead_mse_vs_hfss",
                    "multihead_mse_vs_optimized",
                ]
            ]
            .mean()
            .to_string(),
            flush=True,
        )


if __name__ == "__main__":
    main()
