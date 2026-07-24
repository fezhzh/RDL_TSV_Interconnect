# -*- coding: utf-8 -*-
"""Train the v11 ADS long-chain model with a 7-parameter pi circuit.

Run this file directly in VS Code. No command-line arguments are required.

Flow:
1. Use ADS single-device S-parameters and optimize one shared 7-parameter
   connection circuit per full v11 cascade sample. The same optimized circuit
   is inserted at all 12 connection positions.
2. Train seven independent 9->30->30->20->1 networks from structure geometry
   to the optimized shared circuit parameters.
3. Initialize a final model from those seven networks, then expand each
   20-node parameter layer into 12 connection-position heads and continue
   training with S11/S21 magnitude and wrapped phase loss.
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
import torch.nn as nn
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, TensorDataset


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v11_base.py"
RUN_LABEL = "ads_v08circuit_shared_to_multihead12_lhs150_50_connection2"
DATASET_NAME = "LHS150_50_Connection2"
TARGET_DESIGN_NAME = "TSV_RDL"
CONNECTION_COUNT = 12
DATASET_ROOT = THIS_DIR.parents[2] / "HFSS_sim" / DATASET_NAME
TRAIN_TARGET_CSV = DATASET_ROOT / "train" / f"{TARGET_DESIGN_NAME}_variations_record.csv"
TRAIN_TARGET_SNP_DIR = DATASET_ROOT / "train" / TARGET_DESIGN_NAME
TEST_TARGET_CSV = DATASET_ROOT / "test" / f"{TARGET_DESIGN_NAME}_variations_record.csv"
TEST_TARGET_SNP_DIR = DATASET_ROOT / "test" / TARGET_DESIGN_NAME
SOURCE_ADS_CACHE_DIR = (
    THIS_DIR.parents[1]
    / "v10_ads_pi_cascade"
    / "results"
    / "ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads"
    / "ads_single_device_cache"
)

V08_PARAM_NAMES = ["Cn1_scale", "Rn1_scale", "Cn2_scale", "Rn2_scale", "Cn3_scale", "Rn3_scale", "Ln1_scale"]
V08_SCALE_FACTORS = np.array([1e-14, 1e3, 1e-14, 1e3, 1e-14, 1.0, 1e-11], dtype=np.float64)
V08_P0 = np.ones(len(V08_PARAM_NAMES), dtype=np.float64)
V08_LOWER = np.full(len(V08_PARAM_NAMES), -1e5, dtype=np.float64)
V08_UPPER = np.full(len(V08_PARAM_NAMES), 1e5, dtype=np.float64)

PARAM_EPOCHS = 120
PARAM_PATIENCE = 30
SPARAM_EPOCHS = 90
SPARAM_PATIENCE = 25
PARAM_LR = 8e-4
SPARAM_LR = 2e-5
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-8
PARAM_ANCHOR_WEIGHT = 0.0
OPT_MAX_NFEV = 120
PRINT_EVERY = 10
OPTIMIZED_V08_NMSE_FILTER_THRESHOLD = 0.30


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SharedV08ParamNet(nn.Module):
    """Seven independent scalar networks: 9->30->30->20->1."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.param_nets = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_dim, 30),
                    nn.Tanh(),
                    nn.Linear(30, 30),
                    nn.Tanh(),
                    nn.Linear(30, 20),
                    nn.Tanh(),
                    nn.Linear(20, 1),
                )
                for name in V08_PARAM_NAMES
            }
        )

    def forward(self, x):
        return torch.cat([self.param_nets[name](x) for name in V08_PARAM_NAMES], dim=1)


class MultiHeadV08ConnectionNet(nn.Module):
    """Seven parameter networks with 12 connection-position heads each."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.element_nets = nn.ModuleDict(
            {
                name: nn.ModuleDict(
                    {
                        "trunk": nn.Sequential(
                            nn.Linear(input_dim, 30),
                            nn.Tanh(),
                            nn.Linear(30, 30),
                            nn.Tanh(),
                        ),
                        "heads": nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(30, 20),
                                    nn.Tanh(),
                                    nn.Linear(20, 1),
                                )
                                for _ in range(CONNECTION_COUNT)
                            ]
                        ),
                    }
                )
                for name in V08_PARAM_NAMES
            }
        )

    def initialize_from_shared(self, shared_model: SharedV08ParamNet) -> None:
        for name in V08_PARAM_NAMES:
            src = shared_model.param_nets[name]
            dst = self.element_nets[name]
            dst["trunk"][0].load_state_dict(src[0].state_dict())
            dst["trunk"][2].load_state_dict(src[2].state_dict())
            for head in dst["heads"]:
                head[0].load_state_dict(src[4].state_dict())
                head[2].load_state_dict(src[6].state_dict())

    def forward(self, x):
        conn_outputs = [[] for _ in range(CONNECTION_COUNT)]
        for name in V08_PARAM_NAMES:
            z = self.element_nets[name]["trunk"](x)
            for conn_idx, head in enumerate(self.element_nets[name]["heads"]):
                conn_outputs[conn_idx].append(head(z))
        return torch.cat([torch.cat(values, dim=1) for values in conn_outputs], dim=1)


def v08_target_columns_shared():
    return list(V08_PARAM_NAMES)


def v08_target_columns_multihead():
    return [f"conn{idx}_{name}" for idx in range(1, CONNECTION_COUNT + 1) for name in V08_PARAM_NAMES]


def repeat_shared_params(shared_params: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(shared_params, dtype=np.float64), CONNECTION_COUNT)


def v08_correction_abcd_np(p, omega):
    cn1, rn1, cn2, rn2, cn3, rn3, ln1 = np.asarray(p, dtype=np.float64) * V08_SCALE_FACTORS
    y1 = 1j * omega * cn1 + 1.0 / (rn1 + 1e-30)
    y2 = 1j * omega * cn2 + 1.0 / (rn2 + 1e-30)
    y3 = 1j * omega * cn3 + 1.0 / (rn3 + 1j * omega * ln1 + 1e-30)
    abcd = np.zeros((len(omega), 2, 2), dtype=np.complex128)
    abcd[:, 0, 0] = 1.0 + y2 / y3
    abcd[:, 0, 1] = 1.0 / y3
    abcd[:, 1, 0] = y1 + y2 + y1 * y2 / y3
    abcd[:, 1, 1] = 1.0 + y1 / y3
    return abcd


def cascade_with_v08_np(base, base_abcds, omega, p_flat):
    p_flat = np.asarray(p_flat, dtype=np.float64)
    if p_flat.size == len(V08_PARAM_NAMES):
        p_all = np.tile(p_flat.reshape(1, -1), (CONNECTION_COUNT, 1))
    else:
        p_all = p_flat.reshape(CONNECTION_COUNT, len(V08_PARAM_NAMES))
    result = np.array(base_abcds[0], copy=True)
    for idx in range(CONNECTION_COUNT):
        result = np.matmul(np.matmul(result, v08_correction_abcd_np(p_all[idx], omega)), base_abcds[idx + 1])
    return result


def residual_v08_shared(p, base, base_abcds, target_s, omega):
    pred_s = base.abcd2s(cascade_with_v08_np(base, base_abcds, omega, p))
    diff = pred_s - target_s
    return np.concatenate([diff.real.ravel(), diff.imag.ravel()])


def v08_correction_abcd_torch(p_all, omega, base):
    scales = torch.tensor(V08_SCALE_FACTORS, dtype=base.REAL_DTYPE, device=p_all.device)
    p = p_all * scales
    cn1, rn1, cn2, rn2, cn3, rn3, ln1 = [p[..., i] for i in range(7)]
    j = torch.complex(
        torch.tensor(0.0, dtype=base.REAL_DTYPE, device=p_all.device),
        torch.tensor(1.0, dtype=base.REAL_DTYPE, device=p_all.device),
    )
    omega_b = omega[None, :]
    y1 = j * omega_b * cn1[:, None] + 1.0 / (rn1[:, None].to(base.COMPLEX_DTYPE) + 1e-30)
    y2 = j * omega_b * cn2[:, None] + 1.0 / (rn2[:, None].to(base.COMPLEX_DTYPE) + 1e-30)
    y3 = j * omega_b * cn3[:, None] + 1.0 / (
        rn3[:, None].to(base.COMPLEX_DTYPE) + j * omega_b * ln1[:, None] + 1e-30
    )
    abcd = torch.zeros((p_all.shape[0], len(omega), 2, 2), dtype=base.COMPLEX_DTYPE, device=p_all.device)
    abcd[:, :, 0, 0] = 1.0 + y2 / y3
    abcd[:, :, 0, 1] = 1.0 / y3
    abcd[:, :, 1, 0] = y1 + y2 + y1 * y2 / y3
    abcd[:, :, 1, 1] = 1.0 + y1 / y3
    return abcd


def cascade_with_v08_torch(base, base_abcds, p_all, omega):
    result = base_abcds[:, 0]
    for idx in range(CONNECTION_COUNT):
        corr = v08_correction_abcd_torch(p_all[:, idx, :], omega, base)
        result = torch.matmul(torch.matmul(result, corr), base_abcds[:, idx + 1])
    return result


def write_blocked_validation(base, message: str) -> None:
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readiness_rows = [
        {
            "item": "train full-chain target CSV",
            "path": str(TRAIN_TARGET_CSV),
            "exists": bool(TRAIN_TARGET_CSV.exists()),
            "count": int(len(pd.read_csv(TRAIN_TARGET_CSV, encoding="utf-8-sig"))) if TRAIN_TARGET_CSV.exists() else 0,
            "role": "training target",
        },
        {
            "item": "train full-chain target S-parameters",
            "path": str(TRAIN_TARGET_SNP_DIR),
            "exists": bool(TRAIN_TARGET_SNP_DIR.exists()),
            "count": int(len(list(TRAIN_TARGET_SNP_DIR.glob("dut*.s2p")))) if TRAIN_TARGET_SNP_DIR.exists() else 0,
            "role": "training target",
        },
        {
            "item": "test full-chain target CSV",
            "path": str(TEST_TARGET_CSV),
            "exists": bool(TEST_TARGET_CSV.exists()),
            "count": int(len(pd.read_csv(TEST_TARGET_CSV, encoding="utf-8-sig"))) if TEST_TARGET_CSV.exists() else 0,
            "role": "test target",
        },
        {
            "item": "test full-chain target S-parameters",
            "path": str(TEST_TARGET_SNP_DIR),
            "exists": bool(TEST_TARGET_SNP_DIR.exists()),
            "count": int(len(list(TEST_TARGET_SNP_DIR.glob("dut*.s2p")))) if TEST_TARGET_SNP_DIR.exists() else 0,
            "role": "test target",
        },
        {
            "item": "reusable ADS single-device cache",
            "path": str(SOURCE_ADS_CACHE_DIR),
            "exists": bool(SOURCE_ADS_CACHE_DIR.exists()),
            "count": int(len(list(SOURCE_ADS_CACHE_DIR.glob("*.s2p")))) if SOURCE_ADS_CACHE_DIR.exists() else 0,
            "role": "ADS single-device source",
        },
        {
            "item": "LHS400_Connection2 calibration data",
            "path": str(base.PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train"),
            "exists": bool((base.PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train").exists()),
            "count": int(len(list((base.PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train").glob("*_variations_record.csv")))),
            "role": "ADS calibration reference",
        },
    ]
    readiness = pd.DataFrame(readiness_rows)
    readiness_csv = base.OUTPUT_DIR / "data_readiness_summary.csv"
    readiness.to_csv(readiness_csv, index=False, encoding="utf-8-sig")

    fig, ax = base.plt.subplots(figsize=(11, 5), dpi=150)
    colors = ["#16a34a" if exists else "#dc2626" for exists in readiness["exists"]]
    labels = [item.replace(" ", "\n") for item in readiness["item"]]
    ax.bar(range(len(readiness)), readiness["count"], color=colors)
    ax.set_xticks(range(len(readiness)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Available file or row count")
    ax.set_title("v11 Training Data Readiness")
    ax.grid(axis="y", alpha=0.25)
    for idx, row in readiness.iterrows():
        ax.text(idx, row["count"], "OK" if row["exists"] else "MISSING", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    readiness_plot = base.OUTPUT_DIR / "data_readiness_plot.png"
    fig.savefig(readiness_plot)
    base.plt.close(fig)

    report = {
        "run_label": RUN_LABEL,
        "status": "blocked",
        "reason": message,
        "train_target_csv": str(TRAIN_TARGET_CSV),
        "train_target_snp_dir": str(TRAIN_TARGET_SNP_DIR),
        "test_target_csv": str(TEST_TARGET_CSV),
        "test_target_snp_dir": str(TEST_TARGET_SNP_DIR),
        "device_sequence": base.DEVICE_SEQUENCE,
        "connection_count": CONNECTION_COUNT,
        "ads_cache_dir": str(SOURCE_ADS_CACHE_DIR),
        "model_report": str(base.OUTPUT_DIR / "model_report.md"),
        "data_readiness_summary": str(readiness_csv),
        "data_readiness_plot": str(readiness_plot),
    }
    (base.OUTPUT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (base.OUTPUT_DIR / "model_report.md").write_text(
        "\n".join(
            [
                "# v11 Model Training Report",
                "",
                "## Status",
                "",
                "Training did not start because required data is missing.",
                "",
                "## Requested Model",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Structure: `{'-'.join(base.DEVICE_SEQUENCE)}`",
                f"- Device blocks: `{len(base.DEVICE_SEQUENCE)}`",
                f"- Connection positions: `{CONNECTION_COUNT}`",
                "- Connection circuit: 7-parameter Appendix-1 pi circuit.",
                "- Neural network: seven `9->30->30->20->1` scalar pretrain networks expanded to 12 connection heads.",
                "",
                "## Data Readiness",
                "",
                base.dataframe_to_markdown(readiness),
                "",
                "## Training Diagnosis",
                "",
                "- Appendix 6 requests `LHS150_50_Connection2/train` for training and `test` for model testing.",
                "- The model can train only when both target CSV/S-parameter folders and ADS single-device cache are available.",
                "",
                "## Generated Artifacts",
                "",
                f"- Data readiness CSV: `{readiness_csv}`",
                f"- Data readiness plot: `{readiness_plot}`",
                f"- Machine-readable report: `{base.OUTPUT_DIR / 'training_report.json'}`",
            ]
        ),
        encoding="utf-8",
    )
    (base.OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v11 Flow Validation",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Status: blocked before training",
                f"- Reason: {message}",
                f"- Train target CSV: `{TRAIN_TARGET_CSV}`",
                f"- Train target S-parameter directory: `{TRAIN_TARGET_SNP_DIR}`",
                f"- Test target CSV: `{TEST_TARGET_CSV}`",
                f"- Test target S-parameter directory: `{TEST_TARGET_SNP_DIR}`",
                f"- V11 device sequence: `{'-'.join(base.DEVICE_SEQUENCE)}`",
                f"- Connection count: `{CONNECTION_COUNT}`",
                "",
                "## Flow Assessment",
                "",
                "- The ADS single-device calibration step can use `LHS400_Connection2/train/RDL` and `LHS400_Connection2/train/TSV`.",
                "- Appendix 6 target data comes from `LHS150_50_Connection2/train|test/TSV_RDL`.",
                "- The training entry reuses the existing v10 ADS single-device cache for the same LHS150_50 samples.",
                "",
                "## Artifacts",
                "",
                f"- Model report: `{base.OUTPUT_DIR / 'model_report.md'}`",
                f"- Data readiness summary: `{readiness_csv}`",
                f"- Data readiness plot: `{readiness_plot}`",
            ]
        ),
        encoding="utf-8",
    )


def collect_v11_samples(base) -> pd.DataFrame:
    required_paths = [TRAIN_TARGET_CSV, TRAIN_TARGET_SNP_DIR, TEST_TARGET_CSV, TEST_TARGET_SNP_DIR, SOURCE_ADS_CACHE_DIR]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        message = "Missing required v11 training inputs: " + "; ".join(missing_paths)
        write_blocked_validation(base, message)
        raise FileNotFoundError(message)

    rows = []
    for split, csv_path, snp_dir in [
        ("train", TRAIN_TARGET_CSV, TRAIN_TARGET_SNP_DIR),
        ("test", TEST_TARGET_CSV, TEST_TARGET_SNP_DIR),
    ]:
        df = pd.read_csv(csv_path, encoding="utf-8-sig").sort_values("dut_index").reset_index(drop=True)
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["dut_index"] = int(rec["dut_index"])
            if "t_tmrdl" in rec and "h_tmrdl" not in rec:
                rec["h_tmrdl"] = float(rec.pop("t_tmrdl"))
            if "t_bsmrdl" in rec and "h_bsmrdl" not in rec:
                rec["h_bsmrdl"] = float(rec.pop("t_bsmrdl"))
            rec["split"] = split
            rec["source_root"] = DATASET_NAME
            rec["sample_id"] = f"{DATASET_NAME}_{split}_dut{rec['dut_index']}"
            rec["file"] = str(rec.get("file", f"dut{rec['dut_index']}.s2p"))
            rec["snp_path"] = snp_dir / rec["file"]
            rows.append(rec)

    out = pd.DataFrame(rows)
    missing = [col for col in base.STRUCTURE_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"{TARGET_DESIGN_NAME} sample table is missing columns: {missing}")
    missing_files = [str(path) for path in out["snp_path"] if not Path(path).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing v11 full-chain S-parameter files: {missing_files[:5]}")
    if not out["split"].eq("val").any():
        train_idx = out.index[out["split"].eq("train")]
        out.loc[train_idx[-max(1, len(train_idx) // 10) :], "split"] = "val"
    return out.reset_index(drop=True)


def optimize_v08_shared_targets(base, dut_df: pd.DataFrame, sim):
    rows = []
    target_rows = []
    omega = 2.0 * np.pi * sim.freq_hz
    opt_dir = base.OUTPUT_DIR / "v08_shared_sample_optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)

    for i, row in dut_df.iterrows():
        sample_id = str(row["sample_id"])
        json_path = opt_dir / f"{sample_id}_v08_shared_optimization.json"
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        direct_mse = float(np.mean(np.abs(direct_s - sim.target_s[i]) ** 2))
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            p = np.asarray(data["p_shared"], dtype=np.float64)
            nfev = int(data["nfev"])
            success = bool(data["success"])
        else:
            res = least_squares(
                residual_v08_shared,
                V08_P0,
                args=(base, sim.base_abcds[i], sim.target_s[i], omega),
                bounds=(V08_LOWER, V08_UPPER),
                max_nfev=OPT_MAX_NFEV,
            )
            p = res.x
            nfev = int(res.nfev)
            success = bool(res.success)
            json_path.write_text(
                json.dumps(
                    {"sample_id": sample_id, "p_shared": p.tolist(), "nfev": nfev, "success": success},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        opt_s = base.abcd2s(cascade_with_v08_np(base, sim.base_abcds[i], omega, p))
        opt_mse = float(np.mean(np.abs(opt_s - sim.target_s[i]) ** 2))
        rows.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "file": row["file"],
                "dut_index": int(row["dut_index"]),
                "direct_mse": direct_mse,
                "optimized_v08_shared_mse": opt_mse,
                "direct_nmse_s11_s21_ri": base.nmse_s11_s21_real_imag(sim.target_s[i], direct_s),
                "optimized_v08_shared_nmse_s11_s21_ri": base.nmse_s11_s21_real_imag(sim.target_s[i], opt_s),
                "nfev": nfev,
                "success": success,
            }
        )
        flat_row = {"sample_id": sample_id, "split": row["split"]}
        for name in base.STRUCTURE_COLUMNS:
            flat_row[name] = float(row[name])
        for param_idx, name in enumerate(V08_PARAM_NAMES):
            flat_row[name] = float(p[param_idx])
        target_rows.append(flat_row)
        print(f"[opt-v08-shared] {i + 1}/{len(dut_df)} {sample_id}: direct={direct_mse:.3e}, opt={opt_mse:.3e}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(target_rows)


def apply_optimized_v08_filter(dut_df: pd.DataFrame, shared_targets: pd.DataFrame, opt_summary: pd.DataFrame):
    """Exclude samples with poor per-sample shared-circuit optimization before NN training."""

    bad = (
        opt_summary[opt_summary["optimized_v08_shared_nmse_s11_s21_ri"].gt(OPTIMIZED_V08_NMSE_FILTER_THRESHOLD)]
        .sort_values("optimized_v08_shared_nmse_s11_s21_ri", ascending=False)
        .reset_index(drop=True)
    )
    excluded_ids = set(bad["sample_id"])
    dut_filtered = dut_df.copy()
    targets_filtered = shared_targets.copy()
    for idx, row in dut_filtered.iterrows():
        if row["sample_id"] in excluded_ids:
            dut_filtered.at[idx, "split"] = f"excluded_opt_{row['split']}"
    for idx, row in targets_filtered.iterrows():
        if row["sample_id"] in excluded_ids:
            targets_filtered.at[idx, "split"] = f"excluded_opt_{row['split']}"
    print(
        f"[filter] optimized_v08_shared_nmse_s11_s21_ri > {OPTIMIZED_V08_NMSE_FILTER_THRESHOLD}: "
        f"excluded={len(excluded_ids)}",
        flush=True,
    )
    return dut_filtered, targets_filtered, bad


def train_param_model(base, model, x_norm, y_norm, masks, device):
    train_ds = TensorDataset(
        torch.tensor(x_norm[masks["train"]], dtype=base.REAL_DTYPE),
        torch.tensor(y_norm[masks["train"]], dtype=base.REAL_DTYPE),
    )
    val_x = torch.tensor(x_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    val_y = torch.tensor(y_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAM_LR, weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy(model.state_dict())
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
        rows.append({"stage": "shared_param_pretrain", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[param-v08-shared] epoch={epoch}, train={train_loss:.3e}, val={val_loss:.3e}", flush=True)
        if stale >= PARAM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def train_sparam_model(base, model, arrays, device):
    x_norm, y_norm, masks, y_mean, y_std, sim = arrays
    train_idx = np.where(masks["train"])[0]
    train_ds = TensorDataset(
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(x_norm[train_idx], dtype=base.REAL_DTYPE),
        torch.tensor(y_norm[train_idx], dtype=base.REAL_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_x = torch.tensor(x_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    val_y = torch.tensor(y_norm[masks["val"]], dtype=base.REAL_DTYPE, device=device)
    val_base = torch.tensor(sim.base_abcds[masks["val"]], dtype=base.COMPLEX_DTYPE, device=device)
    val_target = torch.tensor(sim.target_s[masks["val"]], dtype=base.COMPLEX_DTYPE, device=device)
    omega = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPARAM_LR, weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_val = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, SPARAM_EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for idx_b, xb, yb in loader:
            idx_np = idx_b.numpy()
            xb = xb.to(device)
            yb = yb.to(device)
            base_b = torch.tensor(sim.base_abcds[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            target_b = torch.tensor(sim.target_s[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(xb)
            p_flat = base.denormalize_params(pred_norm, y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, CONNECTION_COUNT, len(V08_PARAM_NAMES))
            pred_s = base.abcd2s_torch(cascade_with_v08_torch(base, base_b, p_all, omega))
            loss_s = base.s11_s21_mag_phase_loss_torch(pred_s, target_b)
            loss_anchor = torch.mean((pred_norm - yb) ** 2)
            loss = loss_s + PARAM_ANCHOR_WEIGHT * loss_anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss_s.detach().cpu()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_norm = model(val_x)
            val_p = base.denormalize_params(val_norm, y_mean_t, y_std_t).reshape(-1, CONNECTION_COUNT, len(V08_PARAM_NAMES))
            val_s = base.abcd2s_torch(cascade_with_v08_torch(base, val_base, val_p, omega))
            val_s_loss = base.s11_s21_mag_phase_loss_torch(val_s, val_target).item()
            val_anchor = torch.mean((val_norm - val_y) ** 2).item()
        train_s_loss = total / max(seen, 1)
        rows.append(
            {
                "stage": "multihead_sparam_finetune",
                "epoch": epoch,
                "train_s_loss": train_s_loss,
                "val_s_loss": val_s_loss,
                "val_anchor_loss": val_anchor,
            }
        )
        if val_s_loss < best_val:
            best_val = val_s_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[sparam-v08-multihead] epoch={epoch}, train_s={train_s_loss:.3e}, val_s={val_s_loss:.3e}", flush=True)
        if stale >= SPARAM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def evaluate_model(base, model, dut_df, arrays, device):
    x_norm, _, _, y_mean, y_std, sim = arrays
    omega_t = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    metric_rows = []
    pred_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dut_df), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(dut_df))
            x_b = torch.tensor(x_norm[start:stop], dtype=base.REAL_DTYPE, device=device)
            base_b = torch.tensor(sim.base_abcds[start:stop], dtype=base.COMPLEX_DTYPE, device=device)
            p_flat = base.denormalize_params(model(x_b), y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, CONNECTION_COUNT, len(V08_PARAM_NAMES))
            pred_s = base.abcd2s_torch(cascade_with_v08_torch(base, base_b, p_all, omega_t)).cpu().numpy()
            pred_params = p_flat.cpu().numpy()
            for local_i in range(stop - start):
                i = start + local_i
                target = sim.target_s[i]
                direct = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
                row = dut_df.iloc[i]
                metric_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "file": row["file"],
                        "dut_index": int(row["dut_index"]),
                        "direct_mse_vs_target": float(np.mean(np.abs(direct - target) ** 2)),
                        "v08_nn_mse_vs_target": float(np.mean(np.abs(pred_s[local_i] - target) ** 2)),
                        "direct_nmse_s11_s21_ri": base.nmse_s11_s21_real_imag(target, direct),
                        "v08_nn_nmse_s11_s21_ri": base.nmse_s11_s21_real_imag(target, pred_s[local_i]),
                        "direct_mag_phase_mse_s11_s21": base.s11_s21_mag_phase_mse_np(target, direct),
                        "v08_nn_mag_phase_mse_s11_s21": base.s11_s21_mag_phase_mse_np(target, pred_s[local_i]),
                        "v08_nn_s11_db_mae": float(np.mean(np.abs(base.db20(pred_s[local_i, :, 0, 0]) - base.db20(target[:, 0, 0])))),
                        "v08_nn_s21_db_mae": float(np.mean(np.abs(base.db20(pred_s[local_i, :, 1, 0]) - base.db20(target[:, 1, 0])))),
                    }
                )
                pred_row = {"sample_id": row["sample_id"], "split": row["split"]}
                for col_idx, col_name in enumerate(v08_target_columns_multihead()):
                    pred_row[f"pred_{col_name}"] = float(pred_params[local_i, col_idx])
                pred_rows.append(pred_row)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def summarize_metrics(metrics: pd.DataFrame):
    return (
        metrics.groupby("split", as_index=False)
        .agg(
            count=("sample_id", "count"),
            direct_mse_mean=("direct_mse_vs_target", "mean"),
            v08_nn_mse_mean=("v08_nn_mse_vs_target", "mean"),
            v08_nn_mse_median=("v08_nn_mse_vs_target", "median"),
            direct_nmse_s11_s21_ri_mean=("direct_nmse_s11_s21_ri", "mean"),
            v08_nn_nmse_s11_s21_ri_mean=("v08_nn_nmse_s11_s21_ri", "mean"),
            v08_nn_nmse_s11_s21_ri_median=("v08_nn_nmse_s11_s21_ri", "median"),
            direct_mag_phase_mse_s11_s21_mean=("direct_mag_phase_mse_s11_s21", "mean"),
            v08_nn_mag_phase_mse_s11_s21_mean=("v08_nn_mag_phase_mse_s11_s21", "mean"),
            v08_nn_s11_db_mae_mean=("v08_nn_s11_db_mae", "mean"),
            v08_nn_s21_db_mae_mean=("v08_nn_s21_db_mae", "mean"),
        )
        .sort_values("split")
    )


def save_comparison_plots(base, model, dut_df, arrays, metrics, shared_targets, device):
    x_norm, _, _, y_mean, y_std, sim = arrays
    plot_dir = base.OUTPUT_DIR / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    candidate_df = metrics[metrics["split"].eq(base.PLOT_SPLIT)].copy()
    worst_df = candidate_df.sort_values("v08_nn_nmse_s11_s21_ri", ascending=False).head(base.PLOT_WORST_SAMPLES)
    random_n = min(base.PLOT_RANDOM_SAMPLES, len(candidate_df))
    random_df = candidate_df.sample(n=random_n, random_state=base.PLOT_RANDOM_SEED) if random_n else candidate_df
    groups = [("random_test", random_df), ("worst_test", worst_df)]
    omega = 2.0 * np.pi * sim.freq_hz
    omega_t = torch.tensor(omega, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    freq_ghz = sim.freq_hz / 1e9
    saved = []
    model.eval()
    for group_name, plot_df in groups:
        group_dir = plot_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for _, metric in plot_df.iterrows():
            idx = int(dut_df.index[dut_df["sample_id"].eq(metric["sample_id"])][0])
            x_b = torch.tensor(x_norm[idx : idx + 1], dtype=base.REAL_DTYPE, device=device)
            base_b = torch.tensor(sim.base_abcds[idx : idx + 1], dtype=base.COMPLEX_DTYPE, device=device)
            with torch.no_grad():
                p_flat = base.denormalize_params(model(x_b), y_mean_t, y_std_t)
                p_all = p_flat.reshape(-1, CONNECTION_COUNT, len(V08_PARAM_NAMES))
                pred_s = base.abcd2s_torch(cascade_with_v08_torch(base, base_b, p_all, omega_t)).cpu().numpy()[0]
            direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[idx])))
            opt_p = shared_targets.loc[shared_targets["sample_id"].eq(metric["sample_id"]), V08_PARAM_NAMES].iloc[0].to_numpy(dtype=np.float64)
            optimized_s = base.abcd2s(cascade_with_v08_np(base, sim.base_abcds[idx], omega, opt_p))
            target_s = sim.target_s[idx]
            fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
            fig.suptitle(
                f"{group_name} | {metric['sample_id']} | direct={base.nmse_s11_s21_real_imag(target_s, direct_s):.3e} | "
                f"opt={base.nmse_s11_s21_real_imag(target_s, optimized_s):.3e} | model={base.nmse_s11_s21_real_imag(target_s, pred_s):.3e}",
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
            for ax, (m, n, label, component_fn) in zip(axes.ravel(), specs):
                ax.plot(freq_ghz, component_fn(target_s[:, m, n]), label="HFSS simulation", color="black", linewidth=1.8)
                ax.plot(freq_ghz, component_fn(direct_s[:, m, n]), label="ADS direct cascade", color="#64748b", linestyle=":")
                ax.plot(freq_ghz, component_fn(optimized_s[:, m, n]), label="Optimized shared v08", color="#16a34a", linestyle="--")
                ax.plot(freq_ghz, component_fn(pred_s[:, m, n]), label="V08-circuit NN", color="#dc2626", linestyle="-.")
                ax.set_title(label)
                ax.set_xlabel("Frequency (GHz)")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            out_path = group_dir / f"{metric['sample_id']}_comparison.png"
            fig.savefig(out_path)
            base.plt.close(fig)
            saved.append(str(out_path))
    return plot_dir, saved


def save_checkpoint(base, model, path, sim, settings, x_mean, x_std, y_mean, y_std, target_columns, stage):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": "v11_ads_v08_circuit_shared_to_multihead12",
                "stage": stage,
                "feature_columns": base.STRUCTURE_COLUMNS,
                "target_columns": target_columns,
                "v08_param_names": V08_PARAM_NAMES,
                "v08_scale_factors": V08_SCALE_FACTORS.tolist(),
                "connection_count": CONNECTION_COUNT,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "freq_hz": sim.freq_hz.tolist(),
                "ads_settings": settings,
                "simulation_backend": base.SIMULATION_BACKEND,
                "architecture": "shared 7 scalar nets for pretrain, then 7 parameter nets with 12 30-20-1 heads",
            },
        },
        path,
    )


def run_once(base, settings):
    base.set_seed(base.RANDOM_SEED)
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dut_df = base.collect_samples()
    print(f"Samples: {len(dut_df)}", flush=True)
    sim = base.load_single_device_simulation(dut_df, settings)
    opt_summary, shared_targets = optimize_v08_shared_targets(base, dut_df, sim)
    dut_df, shared_targets, excluded_opt = apply_optimized_v08_filter(dut_df, shared_targets, opt_summary)

    masks = base.split_masks(shared_targets)
    x_raw = shared_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_shared_raw = shared_targets[V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = base.normalize_by_train(x_raw, masks["train"])
    y_shared_norm, y_shared_mean, y_shared_std = base.normalize_by_train(y_shared_raw, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    shared_model = SharedV08ParamNet(input_dim=x_norm.shape[1]).to(dtype=base.REAL_DTYPE, device=device)
    param_history = train_param_model(base, shared_model, x_norm, y_shared_norm, masks, device)
    save_checkpoint(
        base,
        shared_model,
        base.OUTPUT_DIR / "v08_shared_param_pretrain.pt",
        sim,
        settings,
        x_mean,
        x_std,
        y_shared_mean,
        y_shared_std,
        v08_target_columns_shared(),
        "shared_param_pretrain",
    )

    y_multi_raw = np.asarray([repeat_shared_params(row) for row in y_shared_raw], dtype=np.float64)
    y_multi_norm, y_multi_mean, y_multi_std = base.normalize_by_train(y_multi_raw, masks["train"])
    final_model = MultiHeadV08ConnectionNet(input_dim=x_norm.shape[1]).to(dtype=base.REAL_DTYPE, device=device)
    final_model.initialize_from_shared(shared_model)
    arrays = (x_norm, y_multi_norm, masks, y_multi_mean, y_multi_std, sim)
    pretrain_metrics, pretrain_pred = evaluate_model(base, final_model, dut_df, arrays, device)
    pretrain_summary = summarize_metrics(pretrain_metrics)

    sparam_history = train_sparam_model(base, final_model, arrays, device)
    history = pd.concat([param_history, sparam_history], ignore_index=True)
    metrics, pred_params = evaluate_model(base, final_model, dut_df, arrays, device)
    summary = summarize_metrics(metrics)
    plot_dir, plot_files = save_comparison_plots(base, final_model, dut_df, arrays, metrics, shared_targets, device)

    opt_summary.to_csv(base.OUTPUT_DIR / "v08_shared_optimization_summary.csv", index=False, encoding="utf-8-sig")
    shared_targets.to_csv(base.OUTPUT_DIR / "v08_shared_optimized_targets.csv", index=False, encoding="utf-8-sig")
    excluded_opt.to_csv(base.OUTPUT_DIR / "excluded_optimized_v08_shared_samples.csv", index=False, encoding="utf-8-sig")
    history.to_csv(base.OUTPUT_DIR / "v08_training_history.csv", index=False, encoding="utf-8-sig")
    pretrain_metrics.to_csv(base.OUTPUT_DIR / "v08_sparam_metrics_after_shared_param_pretrain.csv", index=False, encoding="utf-8-sig")
    pretrain_summary.to_csv(base.OUTPUT_DIR / "v08_sparam_summary_after_shared_param_pretrain.csv", index=False, encoding="utf-8-sig")
    pretrain_pred.to_csv(base.OUTPUT_DIR / "v08_param_predictions_after_shared_param_pretrain.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(base.OUTPUT_DIR / "v08_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(base.OUTPUT_DIR / "v08_sparam_summary.csv", index=False, encoding="utf-8-sig")
    pred_params.to_csv(base.OUTPUT_DIR / "v08_param_predictions.csv", index=False, encoding="utf-8-sig")
    save_checkpoint(
        base,
        final_model,
        base.OUTPUT_DIR / "v08_connection_multihead_net.pt",
        sim,
        settings,
        x_mean,
        x_std,
        y_multi_mean,
        y_multi_std,
        v08_target_columns_multihead(),
        "multihead_sparam_finetuned",
    )

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(base.OUTPUT_DIR),
        "workflow": [
            "ADS single-device S-parameters are cascaded.",
            f"Each sample optimizes one shared 7-parameter connection circuit used at all {CONNECTION_COUNT} positions.",
            "Seven independent 9->30->30->20->1 networks are trained against the shared optimized parameters.",
            f"The 20-node layer is expanded into {CONNECTION_COUNT} connection heads per parameter and fine-tuned with S11/S21 magnitude and wrapped phase loss.",
        ],
        "samples": int(len(dut_df)),
        "optimized_v08_nmse_filter_threshold": OPTIMIZED_V08_NMSE_FILTER_THRESHOLD,
        "train_count_after_filter": int(masks["train"].sum()),
        "test_count_after_filter": int(masks["test"].sum()),
        "excluded_count": int(len(excluded_opt)),
        "excluded_by_original_split": excluded_opt.groupby("split").size().to_dict() if len(excluded_opt) else {},
        "simulation": sim.simulator_report,
        "v08_param_names": V08_PARAM_NAMES,
        "param_epochs_completed": int(param_history["epoch"].iloc[-1]) if len(param_history) else 0,
        "sparam_epochs_completed": int(sparam_history["epoch"].iloc[-1]) if len(sparam_history) else 0,
        "pretrain_summary": pretrain_summary.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_files,
    }
    (base.OUTPUT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (base.OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v11 ADS V08-Circuit Shared-to-Multihead Validation",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{base.OUTPUT_DIR}`",
                f"- Backend: `{base.SIMULATION_BACKEND}`",
                f"- Samples: {len(dut_df)}",
                "- Optimized circuit: 7-parameter pi circuit from `建模流程.md` Appendix 1.",
                f"- Optimization constraint: one shared 7-parameter circuit per sample, inserted at all {CONNECTION_COUNT} connection positions.",
                f"- Optimized-result filter: exclude samples with `optimized_v08_shared_nmse_s11_s21_ri > {OPTIMIZED_V08_NMSE_FILTER_THRESHOLD}` before NN training.",
                f"- Active train/test after filter: `{int(masks['train'].sum())}` / `{int(masks['test'].sum())}`",
                f"- Excluded samples: `{len(excluded_opt)}`; details in `excluded_optimized_v08_shared_samples.csv`.",
                "- Param pretrain: seven independent `9->30->30->20->1` networks.",
                f"- S-parameter fine-tune: seven parameter networks, each expanded to {CONNECTION_COUNT} `30->20->1` connection heads.",
                "- S-parameter target: `S11`/`S21` magnitude and wrapped phase.",
                f"- Param epochs completed: {report['param_epochs_completed']}",
                f"- S-parameter epochs completed: {report['sparam_epochs_completed']}",
                f"- Comparison plots: `{plot_dir}`",
                "",
                "## Param-Pretrain Summary",
                "",
                base.dataframe_to_markdown(pretrain_summary),
                "",
                "## Summary",
                "",
                base.dataframe_to_markdown(summary),
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Done: {base.OUTPUT_DIR}", flush=True)
    return summary


def main():
    base = load_module(BASE_SCRIPT, "v11_ads_v08_circuit_train_base")

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain" / "results" / RUN_LABEL
    base.ADS_CACHE_DIR = SOURCE_ADS_CACHE_DIR
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0
    base.OPT_MAX_NFEV = OPT_MAX_NFEV
    base.collect_samples = lambda: collect_v11_samples(base)

    sweep = {
        "freq_start_ghz": 0.1,
        "freq_stop_ghz": 100.0,
        "freq_step_ghz": 0.1,
    }
    settings = {
        "calibration_source": "ac_l400_ref2",
        "dataset": DATASET_NAME,
        "target_design": TARGET_DESIGN_NAME,
        "single_device_calibration_dataset": "LHS400_Connection2/train",
        "connection_circuit": "v11_shared7_then_multihead12",
        "rdl_settings": {
            "er_si": 9.8,
            "cond": 5.8e7,
            "tand": 0.005,
            "l_scale": 1.0,
            "w_scale": 0.65,
            "pitch_scale": 1.25,
            "h_tsv_scale": 1.0,
            "h_rdl_scale": 1.0,
            **sweep,
        },
        "tsv_settings": {
            "er_si": 11.9,
            "cond": 5.8e7,
            "tand": 0.005,
            "c1_scale": 1.0,
            "pitch_scale": 1.0,
            "h_tsv_scale": 1.2,
            "d_scale": 1.0,
            **sweep,
        },
    }
    try:
        run_once(base, settings)
    except FileNotFoundError as exc:
        print(f"[blocked] {exc}", flush=True)
        print(f"[blocked] Validation archive: {base.OUTPUT_DIR / 'validation_archive.md'}", flush=True)
        return


if __name__ == "__main__":
    main()
