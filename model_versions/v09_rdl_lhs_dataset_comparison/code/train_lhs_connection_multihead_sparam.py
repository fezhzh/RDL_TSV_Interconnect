# -*- coding: utf-8 -*-
"""Train the multi-head cascade connection model on LHS TSV_RDL data.

Run this file directly in VS Code. No command-line arguments are required.

Data split:
- train: HFSS_sim/LHS100/train/TSV_RDL + LHS200/train/TSV_RDL + LHS400/train/TSV_RDL
- val:   HFSS_sim/LHS100/val/TSV_RDL
- test:  HFSS_sim/LHS100/test/TSV_RDL

The RDL single-device base models are loaded from v09
lhs100_lhs200_lhs400_lhs800. The TSV base model uses the existing v03
single-device S-parameter fine-tuned model.
"""

import copy
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skrf as rf
import torch
from torch.utils.data import DataLoader

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
V09_CODE_DIR = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "code"
SPARAM_CODE_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "code"
for path in [V08_CODE_DIR, V07_CODE_DIR, V03_CODE_DIR, V09_CODE_DIR, SPARAM_CODE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Calc_SP_and_Opt2 as opt2
import fine_tune_connection_network_on_sparams as sfit
import finetune_matlab_rdl_models_on_sparams as rdl_v09
import train_connection_network_multihead_sparam as base
import train_connection_network_params as param_train
import train_single_device_sparam_model as single_device


OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v09_rdl_lhs_dataset_comparison"
    / "results"
    / "connection_multihead_lhs100_200_400_v09_rdl_all_param_pretrain_sparam"
)
CONNECTION_PARAM_CSV = (
    PROJECT_ROOT
    / "model_versions"
    / "v09_rdl_lhs_dataset_comparison"
    / "results"
    / "connection_network_lhs100_200_400_opt2"
    / "connection_network_params.csv"
)

TRAIN_DATASETS = [
    ("LHS100", "train", PROJECT_ROOT / "HFSS_sim" / "LHS100" / "train"),
    ("LHS200", "train", PROJECT_ROOT / "HFSS_sim" / "LHS200" / "train"),
    ("LHS400", "train", PROJECT_ROOT / "HFSS_sim" / "LHS400" / "train"),
]
VAL_DATASETS = [("LHS100", "val", PROJECT_ROOT / "HFSS_sim" / "LHS100" / "val")]
TEST_DATASETS = [("LHS100", "test", PROJECT_ROOT / "HFSS_sim" / "LHS100" / "test")]
CASCADE_DESIGN = "TSV_RDL"

V09_RDL_DATASET = "lhs100_lhs200_lhs400_lhs800"
V09_RDL_MODEL_DIRS = {
    "TMRDL": PROJECT_ROOT
    / "model_versions"
    / "v09_rdl_lhs_dataset_comparison"
    / "results"
    / "sparam_finetuned_models"
    / V09_RDL_DATASET
    / "TMRDL",
    "BSMRDL": PROJECT_ROOT
    / "model_versions"
    / "v09_rdl_lhs_dataset_comparison"
    / "results"
    / "sparam_finetuned_models"
    / V09_RDL_DATASET
    / "BSMRDL",
}
TSV_MODEL_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v03_single_device_sparam_finetune"
    / "results"
    / "single_device_sparam_TSV_mat4_init_sparam_noanchor"
)

# Direct-run training controls.
RANDOM_SEED = 20260706
USE_CUDA_IF_AVAILABLE = True
BATCH_SIZE_SPARAM = 12
BATCH_SIZE_PARAM = 128
PARAM_PRETRAIN_EPOCHS = 1200
PARAM_LR = 8e-4
PATIENCE_PARAM = 180
SPARAM_EPOCHS = 600
SPARAM_LR = 2e-5
WEIGHT_DECAY = 1e-8
PATIENCE_SPARAM = 120
PRINT_EVERY = 20
RUN_PARAM_PRETRAIN = True
PARAM_ANCHOR_WEIGHT = 1e-6
PLOT_DUT_LIMIT = 10
PLOT_SPLIT = "test"
PLOT_SORT_COLUMN = "multihead_mse_vs_hfss"

REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128

STRUCTURE_COLUMNS_LHS = [
    "pitch",
    "r_tsv",
    "h_tsv",
    "l_tmrdl",
    "w_tmrdl",
    "h_tmrdl",
    "l_bsmrdl",
    "w_bsmrdl",
    "h_bsmrdl",
]
CONNECTION_P0 = np.array([0.01, 1000.0, 0.01, 1000.0, 0.01, 0.01, 0.01], dtype=np.float64)
CONNECTION_STD = np.array([50.0, 100.0, 50.0, 100.0, 50.0, 50.0, 50.0], dtype=np.float64)


def apply_base_training_config():
    base.BATCH_SIZE_PARAM = BATCH_SIZE_PARAM
    base.PARAM_PRETRAIN_EPOCHS = PARAM_PRETRAIN_EPOCHS
    base.PARAM_LR = PARAM_LR
    base.PATIENCE_PARAM = PATIENCE_PARAM
    base.BATCH_SIZE_SPARAM = BATCH_SIZE_SPARAM
    base.SPARAM_EPOCHS = SPARAM_EPOCHS
    base.SPARAM_LR = SPARAM_LR
    base.PATIENCE_SPARAM = PATIENCE_SPARAM
    base.WEIGHT_DECAY = WEIGHT_DECAY
    base.PRINT_EVERY = PRINT_EVERY
    base.PARAM_ANCHOR_WEIGHT = PARAM_ANCHOR_WEIGHT


def set_seed(seed):
    base.set_seed(seed)


def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).stem)
    return [int(part) if part.isdigit() else part for part in parts]


def dut_index(path):
    match = re.search(r"dut(\d+)\.s2p$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse DUT index from {path}")
    return int(match.group(1))


def load_variation_table(split_dir):
    csv_path = split_dir / f"{CASCADE_DESIGN}_variations_record.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing variation table: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "dut_index" not in df.columns:
        df = df.copy()
        df["dut_index"] = np.arange(len(df), dtype=int)
    missing = [col for col in STRUCTURE_COLUMNS_LHS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    return df


def collect_split_rows(dataset_specs, split_label):
    rows = []
    for source_root, source_split, split_dir in dataset_specs:
        design_dir = split_dir / CASCADE_DESIGN
        if not design_dir.exists():
            raise FileNotFoundError(f"Missing Snp directory: {design_dir}")
        var_df = load_variation_table(split_dir)
        var_by_dut = {int(row["dut_index"]): row for _, row in var_df.iterrows()}
        for snp_path in sorted(design_dir.glob("dut*.s2p"), key=natural_key):
            idx = dut_index(snp_path)
            if idx not in var_by_dut:
                print(f"[skip] no variation row for {source_root}/{source_split}/{snp_path.name}", flush=True)
                continue
            rec = var_by_dut[idx]
            row = {
                "sample_id": f"{source_root}_{source_split}_dut{idx}",
                "source_root": source_root,
                "source_split": source_split,
                "split": split_label,
                "file": snp_path.name,
                "dut_index": int(idx),
                "variant": "lhs_sparam_only",
                "snp_path": str(snp_path),
            }
            for col in STRUCTURE_COLUMNS_LHS:
                row[col] = float(rec[col])
            rows.append(row)
    return rows


def load_lhs_dataframe():
    rows = []
    rows.extend(collect_split_rows(TRAIN_DATASETS, "train"))
    rows.extend(collect_split_rows(VAL_DATASETS, "val"))
    rows.extend(collect_split_rows(TEST_DATASETS, "test"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No LHS TSV_RDL samples were found.")
    return df.reset_index(drop=True)


def load_v09_rdl_models():
    models = {}
    for device_name, model_dir in V09_RDL_MODEL_DIRS.items():
        checkpoint_path = model_dir / "matlab_param_net_sparam_finetuned.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing v09 RDL model: {checkpoint_path}")
        model = rdl_v09.MatlabMultiParamNet(V09_RDL_DATASET, device_name).to(dtype=REAL_DTYPE)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[device_name] = {"model": model, "metadata": checkpoint["metadata"], "checkpoint": str(checkpoint_path)}
    return models


def load_tsv_model():
    checkpoint_path = TSV_MODEL_DIR / "single_device_sparam_net.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing TSV model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    config = single_device.DEVICE_CONFIGS["TSV"]
    model = single_device.Mat4InitializedDeviceNet(config, metadata).to(dtype=REAL_DTYPE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {"model": model, "metadata": metadata, "checkpoint": str(checkpoint_path)}


def predict_rdl_params(model_info, x_raw):
    with torch.no_grad():
        params = torch.clamp(
            model_info["model"](torch.tensor(np.asarray(x_raw, dtype=np.float64), dtype=REAL_DTYPE)),
            min=1e-30,
        )
    return params.cpu().numpy()


def predict_tsv_params(model_info, x_raw):
    metadata = model_info["metadata"]
    x = np.asarray(x_raw, dtype=np.float64)
    x_norm = (x - np.asarray(metadata["x_mean"], dtype=np.float64)) / np.asarray(metadata["x_std"], dtype=np.float64)
    with torch.no_grad():
        pred_norm = model_info["model"](torch.tensor(x_norm, dtype=REAL_DTYPE))
    y_mean = np.asarray(metadata["y_log_mean"], dtype=np.float64).reshape(1, -1)
    y_std = np.asarray(metadata["y_log_std"], dtype=np.float64).reshape(1, -1)
    return np.exp(np.clip(pred_norm.cpu().numpy() * y_std + y_mean, -40.0, 40.0))


def load_targets_and_freq(dut_df):
    targets = []
    freq = None
    for path in dut_df["snp_path"]:
        nw = rf.Network(str(path))
        if freq is None:
            freq = nw.f
        elif len(freq) != len(nw.f) or not np.allclose(freq, nw.f):
            raise ValueError(f"Frequency grid mismatch: {path}")
        targets.append(nw.s)
    return np.stack(targets, axis=0), freq


def build_base_abcds(dut_df, freq_hz):
    rdl_models = load_v09_rdl_models()
    tsv_model = load_tsv_model()

    top_x = dut_df[["pitch", "l_tmrdl", "w_tmrdl", "h_tmrdl"]].to_numpy(dtype=np.float64)
    bot_x = dut_df[["pitch", "l_bsmrdl", "w_bsmrdl", "h_bsmrdl"]].to_numpy(dtype=np.float64)
    tsv_x = dut_df[["r_tsv", "h_tsv", "pitch"]].to_numpy(dtype=np.float64)

    top_params = predict_rdl_params(rdl_models["TMRDL"], top_x)
    bot_params = predict_rdl_params(rdl_models["BSMRDL"], bot_x)
    tsv_params = predict_tsv_params(tsv_model, tsv_x)

    base_rows = []
    for i, row in dut_df.iterrows():
        cp_tsv = {name: float(tsv_params[i, j]) for j, name in enumerate(single_device.TARGET_PARAMS)}

        s_top = rdl_v09.circuit_params_to_s_np(top_params[i], row["l_tmrdl"] * opt2.DEVICE_LENGTH_SCALE, freq_hz)
        s_bot = rdl_v09.circuit_params_to_s_np(bot_params[i], row["l_bsmrdl"] * opt2.DEVICE_LENGTH_SCALE, freq_hz)
        s_tsv = opt2.calculate_S_parameters(cp_tsv, row["h_tsv"] * opt2.DEVICE_LENGTH_SCALE, freq_hz)
        abcd_top = opt2.s2abcd(s_top)
        abcd_bot = opt2.s2abcd(s_bot)
        abcd_tsv = opt2.s2abcd(s_tsv)
        base_rows.append(
            np.stack([abcd_top, abcd_tsv, abcd_bot, abcd_tsv, abcd_top, abcd_tsv, abcd_bot, abcd_tsv, abcd_top], axis=0)
        )
        if i == 0 or (i + 1) % 100 == 0:
            print(f"Precomputed base cascade {i + 1}/{len(dut_df)}", flush=True)
    return np.stack(base_rows, axis=0), rdl_models, tsv_model


def normalize_train(values, train_mask):
    return base.normalize_train(values, train_mask)


def fallback_connection_targets(n_rows):
    y_raw = np.tile(CONNECTION_P0, param_train.CONNECTION_COUNT).reshape(1, -1)
    y_raw = np.repeat(y_raw, n_rows, axis=0)
    y_mean = np.tile(CONNECTION_P0, param_train.CONNECTION_COUNT)
    y_std = np.tile(CONNECTION_STD, param_train.CONNECTION_COUNT)
    return y_raw, y_mean, y_std, False


def load_connection_targets(dut_df, train_mask):
    if not CONNECTION_PARAM_CSV.exists():
        print(f"[warn] Missing connection-parameter CSV: {CONNECTION_PARAM_CSV}", flush=True)
        return fallback_connection_targets(len(dut_df))

    df = pd.read_csv(CONNECTION_PARAM_CSV, encoding="utf-8-sig")
    required = ["sample_id", "variant", "connection_index"] + param_train.SCALE_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{CONNECTION_PARAM_CSV} is missing columns: {missing}")

    df = df[df["variant"] == "optimized_with_cn3"].copy()
    if df.empty:
        raise ValueError(f"No optimized_with_cn3 rows found in {CONNECTION_PARAM_CSV}")

    rows_by_sample = {}
    for sample_id, group in df.groupby("sample_id", sort=False):
        group = group.sort_values("connection_index")
        if group["connection_index"].astype(int).tolist() != list(range(1, param_train.CONNECTION_COUNT + 1)):
            continue
        values = []
        for _, conn_row in group.iterrows():
            values.extend([float(conn_row[name]) for name in param_train.SCALE_COLUMNS])
        rows_by_sample[str(sample_id)] = values

    missing_samples = [sid for sid in dut_df["sample_id"].astype(str) if sid not in rows_by_sample]
    if missing_samples:
        preview = ", ".join(missing_samples[:5])
        raise ValueError(
            f"Connection target CSV does not cover all LHS samples: missing {len(missing_samples)} samples, e.g. {preview}"
        )

    y_raw = np.asarray([rows_by_sample[str(sid)] for sid in dut_df["sample_id"]], dtype=np.float64)
    y_mean = y_raw[train_mask].mean(axis=0)
    y_std = np.maximum(y_raw[train_mask].std(axis=0), 1e-12)
    print(f"Loaded connection pretrain targets: {CONNECTION_PARAM_CSV}", flush=True)
    return y_raw, y_mean, y_std, True


def build_training_arrays():
    dut_df = load_lhs_dataframe()
    target_s, freq_hz = load_targets_and_freq(dut_df)

    train_mask = dut_df["split"].eq("train").to_numpy()
    val_mask = dut_df["split"].eq("val").to_numpy()
    test_mask = dut_df["split"].eq("test").to_numpy()
    x_raw = dut_df[STRUCTURE_COLUMNS_LHS].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = normalize_train(x_raw, train_mask)

    y_raw, y_mean, y_std, has_connection_targets = load_connection_targets(dut_df, train_mask)
    y_norm = (y_raw - y_mean) / y_std

    base_abcds, rdl_models, tsv_model = build_base_abcds(dut_df, freq_hz)
    dut_indices = np.arange(len(dut_df), dtype=np.int64)
    sample_weights = np.ones(len(dut_df), dtype=np.float64)

    metadata = {
        "model_type": "multihead_connection_net_lhs_param_pretrain_sparam",
        "feature_columns": STRUCTURE_COLUMNS_LHS,
        "target_columns": param_train.TARGET_COLUMNS,
        "scale_columns": param_train.SCALE_COLUMNS,
        "connection_count": param_train.CONNECTION_COUNT,
        "target_variant": param_train.TARGET_VARIANT,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "physical_positive_output": False,
        "physical_lower": np.full(y_raw.shape[1], base.PHYSICAL_LOWER_EPS).tolist(),
        "physical_upper": np.full(y_raw.shape[1], base.PHYSICAL_UPPER_SCALE).tolist(),
        "device_length_scale": opt2.DEVICE_LENGTH_SCALE,
        "freq_hz": freq_hz.tolist(),
        "train_datasets": [(root, split, str(path)) for root, split, path in TRAIN_DATASETS],
        "val_datasets": [(root, split, str(path)) for root, split, path in VAL_DATASETS],
        "test_datasets": [(root, split, str(path)) for root, split, path in TEST_DATASETS],
        "v09_rdl_dataset": V09_RDL_DATASET,
        "v09_rdl_model_dirs": {key: str(value) for key, value in V09_RDL_MODEL_DIRS.items()},
        "v09_rdl_checkpoints": {key: value["checkpoint"] for key, value in rdl_models.items()},
        "tsv_model_checkpoint": tsv_model["checkpoint"],
        "param_pretrain": bool(has_connection_targets),
        "connection_param_csv": str(CONNECTION_PARAM_CSV),
        "connection_initial_p0": CONNECTION_P0.tolist(),
        "connection_initial_std": CONNECTION_STD.tolist(),
    }
    return (
        dut_df,
        x_norm,
        y_norm,
        y_raw,
        train_mask,
        val_mask,
        test_mask,
        metadata,
        base_abcds,
        target_s,
        dut_indices,
        pd.DataFrame(),
        sample_weights,
    )


def initialize_model_for_p0(model):
    for head in model.heads:
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.zeros_(head[-1].bias)


def normalized_output_to_params(pred_norm, metadata, device):
    return base.normalized_output_to_params(pred_norm, metadata, device)


def train_sparam(model, arrays, device):
    _, x_norm, y_norm, _, train_mask, val_mask, _, metadata, base_abcds, target_s, dut_indices, _, sample_weights = arrays
    omega_t = torch.tensor(2.0 * np.pi * np.asarray(metadata["freq_hz"], dtype=np.float64), dtype=REAL_DTYPE, device=device)
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    train_ds = base.SParamDataset(train_indices, x_norm, y_norm, base_abcds, target_s, dut_indices, sample_weights)
    val_ds = base.SParamDataset(val_indices, x_norm, y_norm, base_abcds, target_s, dut_indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_SPARAM, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_SPARAM, shuffle=False)

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
                loss_s = torch.mean(loss_s_each * weight_b / torch.mean(weight_b))
                loss_anchor = torch.mean(loss_anchor_each)
                loss = loss_s + PARAM_ANCHOR_WEIGHT * loss_anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError("Multi-head S-parameter training produced NaN/Inf.")
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
        return total_loss / max(n_seen, 1), total_s / max(n_seen, 1), total_anchor / max(n_seen, 1)

    for epoch in range(1, SPARAM_EPOCHS + 1):
        train_loss, train_s, train_anchor = run_loader(train_loader, training=True)
        with torch.no_grad():
            val_loss, val_s, val_anchor = run_loader(val_loader, training=False)
        scheduler.step(val_loss)
        history.append(
            {
                "stage": "sparam_train",
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
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
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
            print(f"[sparam] early stop: epoch={epoch}, best_val={best_val:.6e}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def split_name(train_mask, val_mask, test_mask, idx):
    if train_mask[idx]:
        return "train"
    if val_mask[idx]:
        return "val"
    if test_mask[idx]:
        return "test"
    return "unused"


def predict_metrics_and_params(model, arrays, device):
    dut_df, x_norm, y_norm, y_raw, train_mask, val_mask, test_mask, metadata, base_abcds, target_s, dut_indices, _, _ = arrays
    omega_t = torch.tensor(2.0 * np.pi * np.asarray(metadata["freq_hz"], dtype=np.float64), dtype=REAL_DTYPE, device=device)
    metric_rows = []
    pred_params = []
    pred_s_rows = []
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
        pred_s_rows.append(pred_s_np)
        target_np = target_b.detach().cpu().numpy()
        for local_i in range(stop - start):
            idx = start + local_i
            direct_s = opt2.abcd2s(opt2.cascade_direct(list(base_abcds[idx])))
            row = dut_df.iloc[idx]
            metric_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_root": row["source_root"],
                    "source_split": row["source_split"],
                    "file": row["file"],
                    "dut_index": int(row["dut_index"]),
                    "split": split_name(train_mask, val_mask, test_mask, idx),
                    "direct_mse_vs_hfss": opt2.mse(target_np[local_i], direct_s),
                    "multihead_mse_vs_hfss": opt2.mse(target_np[local_i], pred_s_np[local_i]),
                    "direct_s11_db_mae": float(np.mean(np.abs(db20(direct_s[:, 0, 0]) - db20(target_np[local_i, :, 0, 0])))),
                    "direct_s21_db_mae": float(np.mean(np.abs(db20(direct_s[:, 1, 0]) - db20(target_np[local_i, :, 1, 0])))),
                    "multihead_s11_db_mae": float(
                        np.mean(np.abs(db20(pred_s_np[local_i, :, 0, 0]) - db20(target_np[local_i, :, 0, 0])))
                    ),
                    "multihead_s21_db_mae": float(
                        np.mean(np.abs(db20(pred_s_np[local_i, :, 1, 0]) - db20(target_np[local_i, :, 1, 0])))
                    ),
                }
            )
    pred_df = dut_df[["sample_id", "source_root", "source_split", "file", "dut_index", "variant"] + STRUCTURE_COLUMNS_LHS].copy()
    pred_arr = np.vstack(pred_params)
    extra_cols = {}
    for i, col in enumerate(param_train.TARGET_COLUMNS):
        extra_cols[f"target_{col}"] = y_raw[:, i]
        extra_cols[f"pred_{col}"] = pred_arr[:, i]
    pred_df = pd.concat([pred_df, pd.DataFrame(extra_cols)], axis=1)
    return pd.DataFrame(metric_rows), pred_df, np.vstack(pred_s_rows)


def save_plots(metrics_df, pred_df, pred_s_all, arrays):
    dut_df, _, _, _, _, _, _, metadata, base_abcds, target_s, _, _, _ = arrays
    plot_df = metrics_df if PLOT_SPLIT == "all" else metrics_df[metrics_df["split"] == PLOT_SPLIT]
    plot_df = plot_df.sort_values(PLOT_SORT_COLUMN, ascending=False)
    plot_dir = OUTPUT_DIR / "multihead_sparam_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    freq_ghz = np.asarray(metadata["freq_hz"], dtype=np.float64) / 1e9
    n_plotted = 0
    for _, metric_row in plot_df.iterrows():
        if PLOT_DUT_LIMIT is not None and n_plotted >= PLOT_DUT_LIMIT:
            break
        idx = int(dut_df.index[dut_df["sample_id"] == metric_row["sample_id"]][0])
        direct_s = opt2.abcd2s(opt2.cascade_direct(list(base_abcds[idx])))
        pred_s = pred_s_all[idx]
        hfss_s = target_s[idx]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        fig.suptitle(f"{metric_row['sample_id']} multi-head LHS S-parameter model", x=0.02, y=0.985, ha="left")
        fig.text(
            0.02,
            0.955,
            f"Direct={metric_row['direct_mse_vs_hfss']:.3e} | Multi-head={metric_row['multihead_mse_vs_hfss']:.3e}",
            ha="left",
            va="top",
            fontsize=9,
            color="#475569",
        )
        for ax, (m, n, label) in zip(axes.ravel(), [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]):
            ax.plot(freq_ghz, db20(hfss_s[:, m, n]), label="HFSS", color="black", linewidth=2.0)
            ax.plot(freq_ghz, db20(direct_s[:, m, n]), label="Direct base cascade", color="#64748b", linestyle=":")
            ax.plot(freq_ghz, db20(pred_s[:, m, n]), label="Multi-head NN", color="#16a34a", linestyle="-.")
            ax.set_title(f"{label} magnitude")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.9, wspace=0.2, hspace=0.3)
        fig.savefig(plot_dir / f"{metric_row['sample_id']}_multihead_sparam.png", dpi=150)
        plt.close(fig)
        n_plotted += 1
    return plot_dir, n_plotted


def summarize_metrics(metrics_df):
    rows = []
    for split, group in metrics_df.groupby("split"):
        rows.append(
            {
                "split": split,
                "count": int(len(group)),
                "direct_mse_mean": float(group["direct_mse_vs_hfss"].mean()),
                "multihead_mse_mean": float(group["multihead_mse_vs_hfss"].mean()),
                "direct_mse_median": float(group["direct_mse_vs_hfss"].median()),
                "multihead_mse_median": float(group["multihead_mse_vs_hfss"].median()),
                "multihead_mse_p95": float(group["multihead_mse_vs_hfss"].quantile(0.95)),
                "multihead_mse_max": float(group["multihead_mse_vs_hfss"].max()),
                "multihead_s11_db_mae_mean": float(group["multihead_s11_db_mae"].mean()),
                "multihead_s21_db_mae_mean": float(group["multihead_s21_db_mae"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    set_seed(RANDOM_SEED)
    apply_base_training_config()
    base.REAL_DTYPE = REAL_DTYPE
    base.COMPLEX_DTYPE = COMPLEX_DTYPE
    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)

    arrays = build_training_arrays()
    dut_df, x_norm, _, _, train_mask, val_mask, test_mask, metadata, _, _, _, _, _ = arrays
    print(
        f"Samples: total={len(dut_df)}, train={int(train_mask.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())}",
        flush=True,
    )
    print(f"Input dim={x_norm.shape[1]}, output dim={len(param_train.TARGET_COLUMNS)}", flush=True)

    model = base.MultiHeadConnectionNet(
        input_dim=x_norm.shape[1],
        connection_count=param_train.CONNECTION_COUNT,
        head_dim=len(param_train.SCALE_COLUMNS),
    ).to(dtype=REAL_DTYPE, device=device)
    initialize_model_for_p0(model)

    do_param_pretrain = bool(RUN_PARAM_PRETRAIN and metadata.get("param_pretrain", False))
    if do_param_pretrain:
        print(f"Param pretrain: enabled, targets={CONNECTION_PARAM_CSV}", flush=True)
        param_history = base.train_param_pretrain(model, arrays, device)
    else:
        print("Param pretrain: disabled; connection-parameter targets are unavailable.", flush=True)
        param_history = pd.DataFrame()
    sparam_history = train_sparam(model, arrays, device)
    history_df = pd.concat([param_history, sparam_history], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "model_type": metadata["model_type"],
            "connection_count": param_train.CONNECTION_COUNT,
            "head_dim": len(param_train.SCALE_COLUMNS),
            "param_anchor_weight": PARAM_ANCHOR_WEIGHT,
            "run_param_pretrain": do_param_pretrain,
            "connection_param_csv": str(CONNECTION_PARAM_CSV),
            "source_script": str(Path(__file__).resolve()),
        },
        OUTPUT_DIR / "connection_param_multihead_net.pt",
    )
    history_df.to_csv(OUTPUT_DIR / "multihead_training_history.csv", index=False, encoding="utf-8-sig")

    metrics_df, pred_df, pred_s_all = predict_metrics_and_params(model, arrays, device)
    metrics_df.to_csv(OUTPUT_DIR / "multihead_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUTPUT_DIR / "multihead_param_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(OUTPUT_DIR / "multihead_sparam_summary.csv", index=False, encoding="utf-8-sig")
    plot_dir, n_plots = save_plots(metrics_df, pred_df, pred_s_all, arrays)

    report = {
        "output_dir": str(OUTPUT_DIR),
        "n_total": int(len(dut_df)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "device": str(device),
        "run_param_pretrain": do_param_pretrain,
        "connection_param_csv": str(CONNECTION_PARAM_CSV),
        "param_pretrain_epochs_completed": int(param_history["epoch"].iloc[-1]) if len(param_history) else 0,
        "sparam_epochs_completed": int(sparam_history["epoch"].iloc[-1]) if len(sparam_history) else 0,
        "best_val_s_loss": float(sparam_history["val_s_loss"].min()) if len(sparam_history) else None,
        "plot_dir": str(plot_dir),
        "n_plots": int(n_plots),
        "summary": summary_df.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nTraining complete", flush=True)
    print(f"Model: {OUTPUT_DIR / 'connection_param_multihead_net.pt'}", flush=True)
    print(f"Metrics: {OUTPUT_DIR / 'multihead_sparam_metrics.csv'}", flush=True)
    print(f"Summary: {OUTPUT_DIR / 'multihead_sparam_summary.csv'}", flush=True)
    print(f"Plots: {plot_dir}", flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
