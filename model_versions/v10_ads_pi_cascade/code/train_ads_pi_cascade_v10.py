# -*- coding: utf-8 -*-
"""Train the v10 ADS-driven pi-network cascade model.

Run this file directly in VS Code. No command-line arguments are required.

The normal production flow is ADS -> single-device S-parameters -> pi-network
optimization -> pi-parameter NN -> S-parameter fine-tuning. The default
``development_cached_snp`` backend reuses the current v09/v03 single-device
models to smoke-test the method before ADS workspace paths are configured.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skrf as rf
import torch
import torch.nn as nn
import matplotlib
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RDL_ADS_SIM_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "rdl_ads_sim"
TSV_ADS_SIM_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "tsv_ads_sim"
V09_CODE_DIR = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "code"
V02_CODE_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "code"
for code_dir in [V09_CODE_DIR, V02_CODE_DIR]:
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

import Calc_SP_and_Opt2 as opt2
import train_lhs_connection_multihead_sparam as lhs_base


def load_ads_helper(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ADS helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RDL_ADS = load_ads_helper("v10_rdl_ads_sim", RDL_ADS_SIM_DIR / "ADS_Sim.py")
TSV_ADS = load_ads_helper("v10_tsv_ads_sim", TSV_ADS_SIM_DIR / "ADS_Sim.py")


# Direct-run configuration.
RUN_LABEL = "ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09"
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / RUN_LABEL

# Direct VS Code runs use ADS by default. The current split randomly selects
# 150 samples from the available LHS200/train data for modeling and uses the
# remaining 50 samples for test.
SIMULATION_BACKEND = "ads"  # "ads" or "development_cached_snp"
RANDOM_SEED = 20260707
USE_CUDA_IF_AVAILABLE = True
LHS200_MODEL_COUNT = 150
LHS200_TEST_COUNT = 50
LHS200_RANDOM_SPLIT_SEED = 20260707
USE_MODEL_SET_AS_VALIDATION = True
ADS_DEVICE_LENGTH_SCALE = 0.9

# ADS runner configuration. Keep these as code-defined defaults for VS Code.
ADS_CACHE_DIR = OUTPUT_DIR / "ads_single_device_cache"

# Material/simulation settings passed to ADS. The script records them and can
# sweep them after ADS is connected.
BASE_ADS_SETTINGS = {
    "substrate_thickness_um": 100.0,
    "substrate_er": 11.9,
    "substrate_loss_tangent": 0.005,
    "metal_thickness_um": 3.0,
    "metal_conductivity_s_per_m": 5.8e7,
}
RUN_MATERIAL_SWEEP = False
MATERIAL_SWEEP_SETTINGS = [
    BASE_ADS_SETTINGS,
    {**BASE_ADS_SETTINGS, "substrate_thickness_um": 80.0},
    {**BASE_ADS_SETTINGS, "substrate_er": 10.8},
    {**BASE_ADS_SETTINGS, "metal_thickness_um": 5.0},
    {**BASE_ADS_SETTINGS, "metal_conductivity_s_per_m": 4.1e7},
]

# Pi network: shunt C left, series R+L, shunt C right.
CONNECTION_COUNT = 8
PI_PARAM_NAMES = ["Cleft_scale", "Rseries_scale", "Lseries_scale", "Cright_scale"]
PI_P0 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
PI_LOWER = np.array([-1e5, -1e5, -1e5, -1e5], dtype=np.float64)
PI_UPPER = np.array([1e5, 1e5, 1e5, 1e5], dtype=np.float64)
PI_SCALE_FACTORS = np.array([1e-14, 1.0, 1e-11, 1e-14], dtype=np.float64)
OPT_MAX_NFEV = 80

# Small defaults are intentional for the first method pass. Increase after the
# ADS backend is connected and the smoke path is stable.
PARAM_EPOCHS = 60
PARAM_PATIENCE = 18
PARAM_LR = 8e-4
SPARAM_EPOCHS = 35
SPARAM_PATIENCE = 12
SPARAM_LR = 2e-5
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-8
PARAM_ANCHOR_WEIGHT = 0.0
PRINT_EVERY = 10
PLOT_SPLIT = "test"
PLOT_RANDOM_SAMPLES = 6
PLOT_WORST_SAMPLES = 6
PLOT_RANDOM_SEED = 20260707

REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
Z_REF = 50.0

STRUCTURE_COLUMNS = [
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


@dataclass
class SimulationBundle:
    base_abcds: np.ndarray
    target_s: np.ndarray
    freq_hz: np.ndarray
    simulator_report: dict


class AdsSimulationRunner:
    def __init__(self, settings: dict[str, object]):
        self.settings = dict(settings)
        self.cache_dir = ADS_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def settings_for_device(self, device_name: str) -> dict[str, float]:
        key = "tsv_settings" if device_name == "TSV" else "rdl_settings"
        if key in self.settings:
            return dict(self.settings[key])
        return dict(self.settings)

    def simulate(self, device_name: str, sample_id: str, row: pd.Series) -> rf.Network:
        if SIMULATION_BACKEND == "development_cached_snp":
            raise RuntimeError("Development backend should not call ADS directly.")

        out_base = self.cache_dir / f"{sample_id}_{device_name}"
        helper = TSV_ADS if device_name == "TSV" else RDL_ADS
        ads_row = scale_structure_for_ads(row)
        out_s2p = helper.simulate_single_device(
            device_name=device_name,
            sample_id=sample_id,
            structure=ads_row,
            ads_settings=self.settings_for_device(device_name),
            output_base=out_base,
            reuse_existing=True,
        )
        return rf.Network(str(out_s2p))


class PiConnectionNet(nn.Module):
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
                for name in PI_PARAM_NAMES
            }
        )

    def forward(self, x):
        conn_outputs = [[] for _ in range(CONNECTION_COUNT)]
        for name in PI_PARAM_NAMES:
            z = self.element_nets[name]["trunk"](x)
            for conn_idx, head in enumerate(self.element_nets[name]["heads"]):
                conn_outputs[conn_idx].append(head(z))
        return torch.cat([torch.cat(values, dim=1) for values in conn_outputs], dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scale_structure_for_ads(row: pd.Series) -> pd.Series:
    ads_row = row.copy()
    for name in ["l_tmrdl", "l_bsmrdl", "h_tsv"]:
        if name in ads_row:
            ads_row[name] = float(ads_row[name]) * ADS_DEVICE_LENGTH_SCALE
    return ads_row
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


def collect_samples() -> pd.DataFrame:
    df = lhs_base.load_lhs_dataframe()
    df = df.sort_values(["split", "source_root", "dut_index"]).reset_index(drop=True)
    lhs200 = df[df["split"].eq("train") & df["source_root"].eq("LHS200")].copy()
    required = LHS200_MODEL_COUNT + LHS200_TEST_COUNT
    if len(lhs200) < required:
        raise ValueError(f"LHS200/train has {len(lhs200)} samples, but {required} are required.")
    shuffled = lhs200.sample(n=required, random_state=LHS200_RANDOM_SPLIT_SEED).reset_index(drop=True)
    model_df = shuffled.head(LHS200_MODEL_COUNT).copy()
    test_df = shuffled.iloc[LHS200_MODEL_COUNT:required].copy()
    model_df = model_df.sort_values("dut_index").reset_index(drop=True)
    test_df = test_df.sort_values("dut_index").reset_index(drop=True)
    model_df["split"] = "train"
    test_df["split"] = "test"
    df = pd.concat([model_df, test_df], ignore_index=True)
    missing = [col for col in STRUCTURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Sample table is missing columns: {missing}")
    return df.reset_index(drop=True)


def load_targets_and_freq(dut_df: pd.DataFrame):
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


def s2abcd(s):
    return opt2.s2abcd(s)


def abcd2s(abcd):
    return opt2.abcd2s(abcd)


def load_single_device_simulation(dut_df: pd.DataFrame, settings: dict[str, float]) -> SimulationBundle:
    target_s, freq_hz = load_targets_and_freq(dut_df)
    if SIMULATION_BACKEND == "development_cached_snp":
        base_abcds, _, _ = lhs_base.build_base_abcds(dut_df, freq_hz)
        return SimulationBundle(
            base_abcds=base_abcds,
            target_s=target_s,
            freq_hz=freq_hz,
            simulator_report={
                "backend": SIMULATION_BACKEND,
                "note": "Used current v09/v03 single-device models to smoke-test the v10 pi workflow.",
                "ads_settings": settings,
            },
        )

    runner = AdsSimulationRunner(settings)
    base_rows = []
    device_sequence = ["TMRDL", "TSV", "BSMRDL", "TSV", "TMRDL", "TSV", "BSMRDL", "TSV", "TMRDL"]
    for i, row in dut_df.iterrows():
        blocks = []
        for device_name in device_sequence:
            nw = runner.simulate(device_name, str(row["sample_id"]), row)
            if len(nw.f) != len(freq_hz) or not np.allclose(nw.f, freq_hz):
                raise ValueError(f"ADS frequency grid mismatch: {row['sample_id']} / {device_name}")
            blocks.append(s2abcd(nw.s))
        base_rows.append(np.stack(blocks, axis=0))
        print(f"ADS single-device simulation {i + 1}/{len(dut_df)}", flush=True)
    return SimulationBundle(
        base_abcds=np.stack(base_rows, axis=0),
        target_s=target_s,
        freq_hz=freq_hz,
        simulator_report={"backend": SIMULATION_BACKEND, "ads_settings": settings, "cache_dir": str(ADS_CACHE_DIR)},
    )


def pi_abcd_one_np(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    cleft, rseries, lseries, cright = np.asarray(p, dtype=np.float64) * PI_SCALE_FACTORS
    y1 = 1j * omega * cleft
    z = rseries + 1j * omega * lseries
    y2 = 1j * omega * cright
    left = np.zeros((len(omega), 2, 2), dtype=np.complex128)
    series = np.zeros_like(left)
    right = np.zeros_like(left)
    left[:, 0, 0] = 1.0
    left[:, 1, 1] = 1.0
    left[:, 1, 0] = y1
    series[:, 0, 0] = 1.0
    series[:, 1, 1] = 1.0
    series[:, 0, 1] = z
    right[:, 0, 0] = 1.0
    right[:, 1, 1] = 1.0
    right[:, 1, 0] = y2
    return np.matmul(np.matmul(left, series), right)


def cascade_with_pi_np(base_abcds: np.ndarray, omega: np.ndarray, p_flat: np.ndarray) -> np.ndarray:
    p_all = np.asarray(p_flat, dtype=np.float64).reshape(CONNECTION_COUNT, len(PI_PARAM_NAMES))
    result = np.array(base_abcds[0], copy=True)
    for idx in range(CONNECTION_COUNT):
        result = np.matmul(np.matmul(result, pi_abcd_one_np(p_all[idx], omega)), base_abcds[idx + 1])
    return result


def residual_pi(p_flat, base_abcds, target_s, omega):
    pred_s = abcd2s(cascade_with_pi_np(base_abcds, omega, p_flat))
    diff = pred_s - target_s
    return np.concatenate([diff.real.ravel(), diff.imag.ravel()])


def s11_s21_real_imag_y(s_params: np.ndarray) -> np.ndarray:
    s11 = s_params[:, 0, 0]
    s21 = s_params[:, 1, 0]
    return np.column_stack([s11.real, s11.imag, s21.real, s21.imag]).ravel()


def nmse_s11_s21_real_imag(y_true_s: np.ndarray, y_pred_s: np.ndarray) -> float:
    y_true = s11_s21_real_imag_y(y_true_s)
    y_pred = s11_s21_real_imag_y(y_pred_s)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(numerator / max(denominator, 1e-30))


def wrapped_phase_diff_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.angle(pred) - np.angle(target)
    return np.arctan2(np.sin(delta), np.cos(delta))


def s11_s21_mag_phase_mse_np(y_true_s: np.ndarray, y_pred_s: np.ndarray) -> float:
    true = np.stack([y_true_s[:, 0, 0], y_true_s[:, 1, 0]], axis=-1)
    pred = np.stack([y_pred_s[:, 0, 0], y_pred_s[:, 1, 0]], axis=-1)
    mag_loss = np.mean((np.abs(pred) - np.abs(true)) ** 2)
    phase_loss = np.mean(wrapped_phase_diff_np(pred, true) ** 2)
    return float(mag_loss + phase_loss)


def s11_s21_mag_phase_loss_torch(pred_s, target_s):
    pred = torch.stack([pred_s[..., 0, 0], pred_s[..., 1, 0]], dim=-1)
    target = torch.stack([target_s[..., 0, 0], target_s[..., 1, 0]], dim=-1)
    mag_loss = torch.mean((torch.abs(pred) - torch.abs(target)) ** 2)
    phase_delta = torch.angle(pred) - torch.angle(target)
    phase_delta = torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta))
    phase_loss = torch.mean(phase_delta**2)
    return mag_loss + phase_loss


def optimize_pi_targets(dut_df: pd.DataFrame, sim: SimulationBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    target_rows = []
    lower = np.tile(PI_LOWER, CONNECTION_COUNT)
    upper = np.tile(PI_UPPER, CONNECTION_COUNT)
    p0 = np.tile(PI_P0, CONNECTION_COUNT)
    omega = 2.0 * np.pi * sim.freq_hz
    opt_dir = OUTPUT_DIR / "pi_sample_optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)

    for i, row in dut_df.iterrows():
        sample_id = str(row["sample_id"])
        json_path = opt_dir / f"{sample_id}_pi_optimization.json"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            p = np.asarray(data["p_flat"], dtype=np.float64)
            direct_mse = float(data["direct_mse"])
            optimized_mse = float(data["optimized_mse"])
            nfev = int(data["nfev"])
            success = bool(data["success"])
            direct_s = abcd2s(opt2.cascade_direct(list(sim.base_abcds[i])))
            optimized_s = abcd2s(cascade_with_pi_np(sim.base_abcds[i], omega, p))
        else:
            direct_s = abcd2s(opt2.cascade_direct(list(sim.base_abcds[i])))
            direct_mse = float(np.mean(np.abs(direct_s - sim.target_s[i]) ** 2))
            res = least_squares(
                residual_pi,
                p0,
                args=(sim.base_abcds[i], sim.target_s[i], omega),
                bounds=(lower, upper),
                max_nfev=OPT_MAX_NFEV,
            )
            p = res.x
            optimized_s = abcd2s(cascade_with_pi_np(sim.base_abcds[i], omega, p))
            optimized_mse = float(np.mean(np.abs(optimized_s - sim.target_s[i]) ** 2))
            nfev = int(res.nfev)
            success = bool(res.success)
            json_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "p_flat": p.tolist(),
                        "direct_mse": direct_mse,
                        "optimized_mse": optimized_mse,
                        "nfev": nfev,
                        "success": success,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        direct_nmse = nmse_s11_s21_real_imag(sim.target_s[i], direct_s)
        optimized_nmse = nmse_s11_s21_real_imag(sim.target_s[i], optimized_s)
        rows.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "file": row["file"],
                "dut_index": int(row["dut_index"]),
                "direct_mse": direct_mse,
                "optimized_pi_mse": optimized_mse,
                "direct_nmse_s11_s21_ri": direct_nmse,
                "optimized_pi_nmse_s11_s21_ri": optimized_nmse,
                "improvement_pct": (direct_mse - optimized_mse) / direct_mse * 100 if direct_mse else np.nan,
                "nmse_improvement_pct": (direct_nmse - optimized_nmse) / direct_nmse * 100 if direct_nmse else np.nan,
                "nfev": nfev,
                "success": success,
            }
        )
        flat_row = {"sample_id": sample_id, "split": row["split"]}
        for name in STRUCTURE_COLUMNS:
            flat_row[name] = float(row[name])
        for conn_idx in range(CONNECTION_COUNT):
            for param_idx, name in enumerate(PI_PARAM_NAMES):
                flat_row[f"conn{conn_idx + 1}_{name}"] = float(p.reshape(CONNECTION_COUNT, -1)[conn_idx, param_idx])
        target_rows.append(flat_row)
        print(
            f"[opt] {i + 1}/{len(dut_df)} {sample_id}: direct={direct_mse:.3e}, pi={optimized_mse:.3e}",
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(target_rows)


def normalize_by_train(values: np.ndarray, train_mask: np.ndarray):
    mean = values[train_mask].mean(axis=0)
    std = np.maximum(values[train_mask].std(axis=0), 1e-12)
    return (values - mean) / std, mean, std


def split_masks(df: pd.DataFrame):
    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("val").to_numpy()
    if USE_MODEL_SET_AS_VALIDATION and not val_mask.any():
        val_mask = train_mask.copy()
    return {
        "train": train_mask,
        "val": val_mask,
        "test": df["split"].eq("test").to_numpy(),
    }


def pi_abcd_torch(p_all, omega):
    scales = torch.tensor(PI_SCALE_FACTORS, dtype=REAL_DTYPE, device=p_all.device)
    p = p_all * scales
    cleft = p[..., 0]
    rseries = p[..., 1]
    lseries = p[..., 2]
    cright = p[..., 3]
    j = torch.complex(
        torch.tensor(0.0, dtype=REAL_DTYPE, device=p_all.device),
        torch.tensor(1.0, dtype=REAL_DTYPE, device=p_all.device),
    )
    y1 = j * omega[None, :] * cleft[:, None]
    z = rseries[:, None].to(COMPLEX_DTYPE) + j * omega[None, :] * lseries[:, None]
    y2 = j * omega[None, :] * cright[:, None]
    out = torch.zeros((p_all.shape[0], len(omega), 2, 2), dtype=COMPLEX_DTYPE, device=p_all.device)
    out[:, :, 0, 0] = 1.0 + z * y2
    out[:, :, 0, 1] = z
    out[:, :, 1, 0] = y1 + y2 + y1 * z * y2
    out[:, :, 1, 1] = 1.0 + y1 * z
    return out


def cascade_with_pi_torch(base_abcds, p_all, omega):
    result = base_abcds[:, 0]
    for idx in range(CONNECTION_COUNT):
        pi = pi_abcd_torch(p_all[:, idx, :], omega)
        result = torch.matmul(torch.matmul(result, pi), base_abcds[:, idx + 1])
    return result


def abcd2s_torch(abcd):
    a = abcd[..., 0, 0]
    b = abcd[..., 0, 1]
    c = abcd[..., 1, 0]
    d = abcd[..., 1, 1]
    denom = a + b / Z_REF + c * Z_REF + d + 1e-30
    s = torch.zeros((*a.shape, 2, 2), dtype=COMPLEX_DTYPE, device=abcd.device)
    s[..., 0, 0] = (a + b / Z_REF - c * Z_REF - d) / denom
    s[..., 0, 1] = 2.0 * (a * d - b * c) / denom
    s[..., 1, 0] = 2.0 / denom
    s[..., 1, 1] = (-a + b / Z_REF - c * Z_REF + d) / denom
    return s


def denormalize_params(pred_norm, y_mean, y_std):
    return pred_norm * y_std + y_mean


def train_param_model(model, x_norm, y_norm, masks, device):
    train_ds = TensorDataset(
        torch.tensor(x_norm[masks["train"]], dtype=REAL_DTYPE),
        torch.tensor(y_norm[masks["train"]], dtype=REAL_DTYPE),
    )
    val_x = torch.tensor(x_norm[masks["val"]], dtype=REAL_DTYPE, device=device)
    val_y = torch.tensor(y_norm[masks["val"]], dtype=REAL_DTYPE, device=device)
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
        rows.append({"stage": "param_pretrain", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[param] epoch={epoch}, train={train_loss:.3e}, val={val_loss:.3e}", flush=True)
        if stale >= PARAM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def train_sparam_model(model, arrays, device):
    x_norm, y_norm, masks, y_mean, y_std, sim = arrays
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["val"])[0]
    train_ds = TensorDataset(
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(x_norm[train_idx], dtype=REAL_DTYPE),
        torch.tensor(y_norm[train_idx], dtype=REAL_DTYPE),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_x = torch.tensor(x_norm[masks["val"]], dtype=REAL_DTYPE, device=device)
    val_y = torch.tensor(y_norm[masks["val"]], dtype=REAL_DTYPE, device=device)
    val_base = torch.tensor(sim.base_abcds[masks["val"]], dtype=COMPLEX_DTYPE, device=device)
    val_target = torch.tensor(sim.target_s[masks["val"]], dtype=COMPLEX_DTYPE, device=device)
    omega = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=REAL_DTYPE, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPARAM_LR, weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
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
            base_b = torch.tensor(sim.base_abcds[idx_np], dtype=COMPLEX_DTYPE, device=device)
            target_b = torch.tensor(sim.target_s[idx_np], dtype=COMPLEX_DTYPE, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(xb)
            p_flat = denormalize_params(pred_norm, y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, CONNECTION_COUNT, len(PI_PARAM_NAMES))
            pred_s = abcd2s_torch(cascade_with_pi_torch(base_b, p_all, omega))
            loss_s = s11_s21_mag_phase_loss_torch(pred_s, target_b)
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
            val_p = denormalize_params(val_norm, y_mean_t, y_std_t).reshape(-1, CONNECTION_COUNT, len(PI_PARAM_NAMES))
            val_s = abcd2s_torch(cascade_with_pi_torch(val_base, val_p, omega))
            val_s_loss = s11_s21_mag_phase_loss_torch(val_s, val_target).item()
            val_anchor = torch.mean((val_norm - val_y) ** 2).item()
        train_s_loss = total / max(seen, 1)
        rows.append(
            {
                "stage": "sparam_finetune",
                "epoch": epoch,
                "train_s_loss": train_s_loss,
                "val_s_loss": val_s_loss,
                "val_anchor_loss": val_anchor,
            }
        )
        if val_s_loss < best_val:
            best_val = val_s_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[sparam] epoch={epoch}, train_s={train_s_loss:.3e}, val_s={val_s_loss:.3e}", flush=True)
        if stale >= SPARAM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def evaluate_model(model, dut_df, arrays, device):
    x_norm, _, masks, y_mean, y_std, sim = arrays
    omega_t = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=REAL_DTYPE, device=device)
    metric_rows = []
    pred_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dut_df), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(dut_df))
            x_b = torch.tensor(x_norm[start:stop], dtype=REAL_DTYPE, device=device)
            base_b = torch.tensor(sim.base_abcds[start:stop], dtype=COMPLEX_DTYPE, device=device)
            p_flat = denormalize_params(model(x_b), y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, CONNECTION_COUNT, len(PI_PARAM_NAMES))
            pred_s = abcd2s_torch(cascade_with_pi_torch(base_b, p_all, omega_t)).cpu().numpy()
            pred_params = p_flat.cpu().numpy()
            for local_i in range(stop - start):
                i = start + local_i
                target = sim.target_s[i]
                direct = abcd2s(opt2.cascade_direct(list(sim.base_abcds[i])))
                row = dut_df.iloc[i]
                metric_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "file": row["file"],
                        "dut_index": int(row["dut_index"]),
                        "direct_mse_vs_target": float(np.mean(np.abs(direct - target) ** 2)),
                        "pi_nn_mse_vs_target": float(np.mean(np.abs(pred_s[local_i] - target) ** 2)),
                        "direct_nmse_s11_s21_ri": nmse_s11_s21_real_imag(target, direct),
                        "pi_nn_nmse_s11_s21_ri": nmse_s11_s21_real_imag(target, pred_s[local_i]),
                        "direct_mag_phase_mse_s11_s21": s11_s21_mag_phase_mse_np(target, direct),
                        "pi_nn_mag_phase_mse_s11_s21": s11_s21_mag_phase_mse_np(target, pred_s[local_i]),
                        "pi_nn_s11_db_mae": float(np.mean(np.abs(db20(pred_s[local_i, :, 0, 0]) - db20(target[:, 0, 0])))),
                        "pi_nn_s21_db_mae": float(np.mean(np.abs(db20(pred_s[local_i, :, 1, 0]) - db20(target[:, 1, 0])))),
                    }
                )
                pred_row = {"sample_id": row["sample_id"], "split": row["split"]}
                for col_idx, col_name in enumerate(pi_target_columns()):
                    pred_row[f"pred_{col_name}"] = float(pred_params[local_i, col_idx])
                pred_rows.append(pred_row)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def save_comparison_plots(model, dut_df, arrays, metrics, device):
    x_norm, y_norm, _, y_mean, y_std, sim = arrays
    sort_column = "pi_nn_nmse_s11_s21_ri" if "pi_nn_nmse_s11_s21_ri" in metrics.columns else "pi_nn_mse_vs_target"
    plot_dir = OUTPUT_DIR / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    candidate_df = metrics[metrics["split"].eq(PLOT_SPLIT)].copy()
    worst_df = candidate_df.sort_values(sort_column, ascending=False).head(PLOT_WORST_SAMPLES)
    random_n = min(PLOT_RANDOM_SAMPLES, len(candidate_df))
    random_df = candidate_df.sample(n=random_n, random_state=PLOT_RANDOM_SEED) if random_n else candidate_df
    plot_groups = [
        ("random_test", random_df),
        ("worst_test", worst_df),
    ]

    omega_t = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=REAL_DTYPE, device=device)
    freq_ghz = sim.freq_hz / 1e9
    saved = []
    model.eval()

    for group_name, plot_df in plot_groups:
        group_dir = plot_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for _, metric in plot_df.iterrows():
            idx = int(dut_df.index[dut_df["sample_id"].eq(metric["sample_id"])][0])
            x_b = torch.tensor(x_norm[idx : idx + 1], dtype=REAL_DTYPE, device=device)
            base_b = torch.tensor(sim.base_abcds[idx : idx + 1], dtype=COMPLEX_DTYPE, device=device)
            with torch.no_grad():
                p_flat = denormalize_params(model(x_b), y_mean_t, y_std_t)
                p_all = p_flat.reshape(-1, CONNECTION_COUNT, len(PI_PARAM_NAMES))
                pred_s = abcd2s_torch(cascade_with_pi_torch(base_b, p_all, omega_t)).cpu().numpy()[0]

            direct_s = abcd2s(opt2.cascade_direct(list(sim.base_abcds[idx])))
            opt_p_flat = y_norm[idx] * y_std + y_mean
            optimized_s = abcd2s(cascade_with_pi_np(sim.base_abcds[idx], 2.0 * np.pi * sim.freq_hz, opt_p_flat))
            target_s = sim.target_s[idx]
            optimized_nmse = nmse_s11_s21_real_imag(target_s, optimized_s)
            direct_nmse = metric.get("direct_nmse_s11_s21_ri", nmse_s11_s21_real_imag(target_s, direct_s))
            model_nmse = metric.get("pi_nn_nmse_s11_s21_ri", nmse_s11_s21_real_imag(target_s, pred_s))
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
            title = (
                f"{group_name} | {metric['sample_id']} | NMSE direct={direct_nmse:.3e} | "
                f"optimized={optimized_nmse:.3e} | pi-NN={model_nmse:.3e}"
            )
            fig.suptitle(title, x=0.02, y=0.985, ha="left")
            plot_specs = [
                (0, 0, "S11 real", np.real),
                (0, 0, "S11 imag", np.imag),
                (1, 0, "S21 real", np.real),
                (1, 0, "S21 imag", np.imag),
            ]
            for ax, (m, n, label, component_fn) in zip(axes.ravel(), plot_specs):
                ax.plot(freq_ghz, component_fn(target_s[:, m, n]), label="HFSS simulation", color="black", linewidth=1.8)
                ax.plot(freq_ghz, component_fn(direct_s[:, m, n]), label="ADS direct cascade", color="#64748b", linestyle=":")
                ax.plot(freq_ghz, component_fn(optimized_s[:, m, n]), label="Optimized pi", color="#16a34a", linestyle="--")
                ax.plot(freq_ghz, component_fn(pred_s[:, m, n]), label="Pi-NN model", color="#dc2626", linestyle="-.")
                ax.set_title(label)
                ax.set_xlabel("Frequency (GHz)")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            out_path = group_dir / f"{metric['sample_id']}_comparison.png"
            fig.savefig(out_path)
            plt.close(fig)
            saved.append(str(out_path))
    return plot_dir, saved


def pi_target_columns():
    return [f"conn{idx}_{name}" for idx in range(1, CONNECTION_COUNT + 1) for name in PI_PARAM_NAMES]


def summarize_metrics(metrics: pd.DataFrame):
    return (
        metrics.groupby("split", as_index=False)
        .agg(
            count=("sample_id", "count"),
            direct_mse_mean=("direct_mse_vs_target", "mean"),
            pi_nn_mse_mean=("pi_nn_mse_vs_target", "mean"),
            pi_nn_mse_median=("pi_nn_mse_vs_target", "median"),
            direct_nmse_s11_s21_ri_mean=("direct_nmse_s11_s21_ri", "mean"),
            pi_nn_nmse_s11_s21_ri_mean=("pi_nn_nmse_s11_s21_ri", "mean"),
            pi_nn_nmse_s11_s21_ri_median=("pi_nn_nmse_s11_s21_ri", "median"),
            direct_mag_phase_mse_s11_s21_mean=("direct_mag_phase_mse_s11_s21", "mean"),
            pi_nn_mag_phase_mse_s11_s21_mean=("pi_nn_mag_phase_mse_s11_s21", "mean"),
            pi_nn_s11_db_mae_mean=("pi_nn_s11_db_mae", "mean"),
            pi_nn_s21_db_mae_mean=("pi_nn_s21_db_mae", "mean"),
        )
        .sort_values("split")
    )


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
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_model_checkpoint(
    model,
    path: Path,
    sim: SimulationBundle,
    settings: dict[str, float],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    stage: str,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": "v10_ads_pi_cascade",
                "stage": stage,
                "simulation_backend": SIMULATION_BACKEND,
                "feature_columns": STRUCTURE_COLUMNS,
                "target_columns": pi_target_columns(),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "pi_param_names": PI_PARAM_NAMES,
                "pi_scale_factors": PI_SCALE_FACTORS.tolist(),
                "pi_lower_bounds": PI_LOWER.tolist(),
                "pi_upper_bounds": PI_UPPER.tolist(),
                "connection_count": CONNECTION_COUNT,
                "freq_hz": sim.freq_hz.tolist(),
                "ads_settings": settings,
                "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
                "lhs200_model_count": LHS200_MODEL_COUNT,
                "lhs200_test_count": LHS200_TEST_COUNT,
                "lhs200_random_split_seed": LHS200_RANDOM_SPLIT_SEED,
                "use_model_set_as_validation": USE_MODEL_SET_AS_VALIDATION,
            },
        },
        path,
    )


def run_once(settings: dict[str, float]):
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dut_df = collect_samples()
    print(f"Samples: {len(dut_df)}", flush=True)
    sim = load_single_device_simulation(dut_df, settings)
    opt_summary, pi_targets = optimize_pi_targets(dut_df, sim)

    masks = split_masks(pi_targets)
    x_raw = pi_targets[STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = pi_targets[pi_target_columns()].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = normalize_by_train(x_raw, masks["train"])
    y_norm, y_mean, y_std = normalize_by_train(y_raw, masks["train"])

    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    model = PiConnectionNet(input_dim=x_norm.shape[1]).to(dtype=REAL_DTYPE, device=device)
    param_history = train_param_model(model, x_norm, y_norm, masks, device)
    arrays = (x_norm, y_norm, masks, y_mean, y_std, sim)
    pretrain_metrics, pretrain_pred_params = evaluate_model(model, dut_df, arrays, device)
    pretrain_summary = summarize_metrics(pretrain_metrics)
    save_model_checkpoint(
        model,
        OUTPUT_DIR / "pi_connection_net_param_pretrain.pt",
        sim,
        settings,
        x_mean,
        x_std,
        y_mean,
        y_std,
        stage="param_pretrain",
    )

    sparam_history = train_sparam_model(model, arrays, device)
    history = pd.concat([param_history, sparam_history], ignore_index=True)
    metrics, pred_params = evaluate_model(model, dut_df, arrays, device)
    summary = summarize_metrics(metrics)
    plot_dir, plot_files = save_comparison_plots(model, dut_df, arrays, metrics, device)

    opt_summary.to_csv(OUTPUT_DIR / "pi_optimization_summary.csv", index=False, encoding="utf-8-sig")
    pi_targets.to_csv(OUTPUT_DIR / "pi_optimized_targets.csv", index=False, encoding="utf-8-sig")
    history.to_csv(OUTPUT_DIR / "pi_training_history.csv", index=False, encoding="utf-8-sig")
    pretrain_metrics.to_csv(OUTPUT_DIR / "pi_sparam_metrics_after_param_pretrain.csv", index=False, encoding="utf-8-sig")
    pretrain_summary.to_csv(OUTPUT_DIR / "pi_sparam_summary_after_param_pretrain.csv", index=False, encoding="utf-8-sig")
    pretrain_pred_params.to_csv(OUTPUT_DIR / "pi_param_predictions_after_param_pretrain.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "pi_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "pi_sparam_summary.csv", index=False, encoding="utf-8-sig")
    pred_params.to_csv(OUTPUT_DIR / "pi_param_predictions.csv", index=False, encoding="utf-8-sig")
    save_model_checkpoint(
        model,
        OUTPUT_DIR / "pi_connection_net.pt",
        sim,
        settings,
        x_mean,
        x_std,
        y_mean,
        y_std,
        stage="sparam_finetuned",
    )
    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(OUTPUT_DIR),
        "workflow": [
            "ADS single-device S-parameters are cascaded and each sample optimizes eight pi-network element sets.",
            "The optimized pi elements are saved as pi_optimized_targets.csv and used to train the param-pretrain model.",
            "The same network is then fine-tuned against S11/S21 magnitude and wrapped phase targets.",
        ],
        "samples": int(len(dut_df)),
        "simulation": sim.simulator_report,
        "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
        "metric_definition": {
            "nmse_s11_s21_ri": "sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)",
            "y": "flattened [real(S11), imag(S11), real(S21), imag(S21)] over all frequency points",
        },
        "optimized_pi_dataset": str(OUTPUT_DIR / "pi_optimized_targets.csv"),
        "pi_scale_bounds": {
            "lower": PI_LOWER.tolist(),
            "upper": PI_UPPER.tolist(),
            "note": "Signed bounds allow negative connection-network element scales.",
        },
        "param_pretrain_checkpoint": str(OUTPUT_DIR / "pi_connection_net_param_pretrain.pt"),
        "sparam_finetuned_checkpoint": str(OUTPUT_DIR / "pi_connection_net.pt"),
        "lhs200_model_count": LHS200_MODEL_COUNT,
        "lhs200_test_count": LHS200_TEST_COUNT,
        "lhs200_random_split_seed": LHS200_RANDOM_SPLIT_SEED,
        "use_model_set_as_validation": USE_MODEL_SET_AS_VALIDATION,
        "param_epochs_completed": int(param_history["epoch"].iloc[-1]) if len(param_history) else 0,
        "sparam_epochs_completed": int(sparam_history["epoch"].iloc[-1]) if len(sparam_history) else 0,
        "pretrain_summary": pretrain_summary.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_files,
        "summary": summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v10 ADS Pi Cascade Validation",
                "",
                f"- Backend: `{SIMULATION_BACKEND}`",
                f"- Samples: {len(dut_df)}",
                f"- Output: `{OUTPUT_DIR}`",
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}` applied to `l_tmrdl`, `l_bsmrdl`, and `h_tsv`",
                f"- Pi optimization rows: {len(opt_summary)}",
                f"- Optimized pi dataset: `{OUTPUT_DIR / 'pi_optimized_targets.csv'}`",
                f"- Pi scale bounds: `{PI_LOWER.tolist()}` to `{PI_UPPER.tolist()}`",
                "- Pi scale sign constraint: signed scales are allowed; values are not constrained to be positive",
                f"- Param-pretrain checkpoint: `{OUTPUT_DIR / 'pi_connection_net_param_pretrain.pt'}`",
                f"- S-parameter finetuned checkpoint: `{OUTPUT_DIR / 'pi_connection_net.pt'}`",
                f"- Param epochs completed: {report['param_epochs_completed']}",
                f"- S-parameter epochs completed: {report['sparam_epochs_completed']}",
                f"- Comparison plots: `{plot_dir}`",
                "- NMSE definition: `sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)`",
                "- NMSE y vector: flattened `S11.real`, `S11.imag`, `S21.real`, and `S21.imag` over all frequencies",
                "",
                "## Workflow",
                "",
                "1. Optimize eight pi-network element sets for each cascaded ADS sample.",
                "2. Train the preliminary model from structure parameters to optimized pi element values.",
                "3. Fine-tune the preliminary model with `S11`/`S21` magnitude and wrapped phase loss.",
                "",
                "## Param-Pretrain Summary",
                "",
                dataframe_to_markdown(pretrain_summary),
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                f"Note: `{SIMULATION_BACKEND}` backend was used for this validation run.",
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)
    return summary


def main():
    if RUN_MATERIAL_SWEEP:
        rows = []
        for settings in MATERIAL_SWEEP_SETTINGS:
            rows.append(run_once(settings).assign(settings=json.dumps(settings, ensure_ascii=False)))
        pd.concat(rows, ignore_index=True).to_csv(OUTPUT_DIR / "material_sweep_summary.csv", index=False, encoding="utf-8-sig")
    else:
        run_once(BASE_ADS_SETTINGS)


if __name__ == "__main__":
    main()
