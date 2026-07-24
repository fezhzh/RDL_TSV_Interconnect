# -*- coding: utf-8 -*-
"""Train the v12 HFSS-equivalent v08 long-chain model.

Run this file directly in VS Code. No command-line arguments are required.

Flow:
1. Read the LHS150_50 full-chain HFSS target samples.
2. Build the 13-device base cascade from HFSS-derived equivalent-circuit
   single-device models.
3. Optimize one shared v08 7-parameter pi circuit per full-chain sample.
4. Train seven shared scalar parameter networks.
5. Expand the shared networks into six learned connection heads mirrored to the
   twelve physical connection positions, then fine-tune with S11/S21 magnitude
   and wrapped phase loss.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
V11_CODE_DIR = PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain" / "code"
V09_FINETUNE_SCRIPT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "code" / "finetune_matlab_rdl_models_on_sparams.py"
RDL_CONNECTION2_SCRIPT = THIS_DIR / "train_rdl_connection2_sparam_model.py"
TSV_CONNECTION2_SCRIPT = THIS_DIR / "train_tsv_connection2_sparam_model.py"
BASE_SCRIPT = V11_CODE_DIR / "train_ads_pi_cascade_v11_base.py"
SHARED_SCRIPT = V11_CODE_DIR / "train_v11_shared7_to_multihead12.py"

RUN_LABEL = "hfss_v08_symmetric_multihead_lhs150_50_connection2_new_rdl_tsv"
VERSION_ROOT = PROJECT_ROOT / "model_versions" / "v12_hfss_v08_multihead_chain"
OUTPUT_DIR = VERSION_ROOT / "results" / RUN_LABEL

RDL_CONNECTION2_CHECKPOINT = VERSION_ROOT / "results" / "rdl_connection2_sparam_model" / "rdl_connection2_sparam_net.pt"
TSV_CONNECTION2_CHECKPOINT = VERSION_ROOT / "results" / "tsv_connection2_sparam_continue" / "tsv_connection2_sparam_continue_net.pt"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "v12_base_helpers")
shared = load_module(SHARED_SCRIPT, "v12_v08_shared_helpers")
v09 = load_module(V09_FINETUNE_SCRIPT, "v12_v09_rdl_helpers")
rdl_connection2 = load_module(RDL_CONNECTION2_SCRIPT, "v12_rdl_connection2_helpers")
tsv_connection2 = load_module(TSV_CONNECTION2_SCRIPT, "v12_tsv_connection2_helpers")


class V12RdlConnection2ParamModel(nn.Module):
    def __init__(self, checkpoint_path: Path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.metadata = checkpoint["metadata"]
        self.model = rdl_connection2.RdlParamNet(input_dim=len(self.metadata["feature_columns"])).to(dtype=base.REAL_DTYPE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def forward(self, x_raw):
        x_mean = torch.tensor(self.metadata["x_mean"], dtype=base.REAL_DTYPE).reshape(1, -1)
        x_std = torch.tensor(self.metadata["x_std"], dtype=base.REAL_DTYPE).reshape(1, -1)
        y_mean = torch.tensor(self.metadata["y_log_mean"], dtype=base.REAL_DTYPE).reshape(1, -1)
        y_std = torch.tensor(self.metadata["y_log_std"], dtype=base.REAL_DTYPE).reshape(1, -1)
        x_norm = (x_raw - x_mean) / torch.clamp(x_std, min=1e-30)
        y_norm = self.model(x_norm)
        return torch.exp(y_norm * y_std + y_mean)


class V12TsvConnection2ParamModel(nn.Module):
    def __init__(self, checkpoint_path: Path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.metadata = checkpoint["metadata"]
        self.model = tsv_connection2.TsvParamNet(input_dim=len(self.metadata["feature_columns"])).to(dtype=base.REAL_DTYPE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def forward(self, x_raw):
        x_mean = torch.tensor(self.metadata["x_mean"], dtype=base.REAL_DTYPE).reshape(1, -1)
        x_std = torch.tensor(self.metadata["x_std"], dtype=base.REAL_DTYPE).reshape(1, -1)
        y_mean = torch.tensor(self.metadata["y_log_mean"], dtype=base.REAL_DTYPE).reshape(1, -1)
        y_std = torch.tensor(self.metadata["y_log_std"], dtype=base.REAL_DTYPE).reshape(1, -1)
        x_norm = (x_raw - x_mean) / torch.clamp(x_std, min=1e-30)
        y_norm = self.model(x_norm)
        return torch.exp(y_norm * y_std + y_mean)


class HfssEquivalentDeviceBackend:
    def __init__(self):
        if not RDL_CONNECTION2_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing v12 RDL Connection2 equivalent-circuit checkpoint: {RDL_CONNECTION2_CHECKPOINT}")
        if not TSV_CONNECTION2_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing v12 TSV Connection2 equivalent-circuit checkpoint: {TSV_CONNECTION2_CHECKPOINT}")

        self.rdl_model = V12RdlConnection2ParamModel(RDL_CONNECTION2_CHECKPOINT).to(dtype=base.REAL_DTYPE).eval()
        self.tsv_model = V12TsvConnection2ParamModel(TSV_CONNECTION2_CHECKPOINT).to(dtype=base.REAL_DTYPE).eval()

    def predict_rdl_params(self, row: pd.Series, device_name: str) -> tuple[np.ndarray, float]:
        if device_name == "TMRDL":
            x = np.array([[row["pitch"], row["l_tmrdl"], row["w_tmrdl"], row["h_tmrdl"]]], dtype=np.float64)
            length_um = float(row["l_tmrdl"])
        elif device_name == "BSMRDL":
            x = np.array([[row["pitch"], row["l_bsmrdl"], row["w_bsmrdl"], row["h_bsmrdl"]]], dtype=np.float64)
            length_um = float(row["l_bsmrdl"])
        else:
            raise ValueError(device_name)
        with torch.no_grad():
            params = self.rdl_model(torch.tensor(x, dtype=base.REAL_DTYPE)).cpu().numpy()[0]
        return params, length_um

    def predict_tsv_params(self, row: pd.Series) -> tuple[np.ndarray, float]:
        x = np.array([[float(row["r_tsv"]), row["h_tsv"], row["pitch"]]], dtype=np.float64)
        with torch.no_grad():
            params = self.tsv_model(torch.tensor(x, dtype=base.REAL_DTYPE)).cpu().numpy()[0]
        return params, float(row["h_tsv"])

    def device_s(self, row: pd.Series, device_name: str, freq_hz: np.ndarray) -> np.ndarray:
        if device_name == "TSV":
            params, length_um = self.predict_tsv_params(row)
        else:
            params, length_um = self.predict_rdl_params(row, device_name)
        return v09.circuit_params_to_s_np(params, length_um, freq_hz)


class SymmetricV08ConnectionNet(shared.MultiHeadV08ConnectionNet):
    """Seven parameter networks with six heads mirrored to twelve connections."""

    def __init__(self, input_dim: int):
        nn.Module.__init__(self)
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
                                for _ in range(6)
                            ]
                        ),
                    }
                )
                for name in shared.V08_PARAM_NAMES
            }
        )

    def initialize_from_shared(self, shared_model: shared.SharedV08ParamNet) -> None:
        for name in shared.V08_PARAM_NAMES:
            src = shared_model.param_nets[name]
            dst = self.element_nets[name]
            dst["trunk"][0].load_state_dict(src[0].state_dict())
            dst["trunk"][2].load_state_dict(src[2].state_dict())
            for head in dst["heads"]:
                head[0].load_state_dict(src[4].state_dict())
                head[2].load_state_dict(src[6].state_dict())

    def forward(self, x):
        unique_outputs = [[] for _ in range(6)]
        for name in shared.V08_PARAM_NAMES:
            z = self.element_nets[name]["trunk"](x)
            for head_idx, head in enumerate(self.element_nets[name]["heads"]):
                unique_outputs[head_idx].append(head(z))
        unique_flat = [torch.cat(values, dim=1) for values in unique_outputs]
        mirrored = unique_flat + list(reversed(unique_flat))
        return torch.cat(mirrored, dim=1)


def load_targets_and_freq(dut_df: pd.DataFrame):
    targets = []
    freq_hz = None
    for path in dut_df["snp_path"]:
        nw = base.rf.Network(str(path))
        if freq_hz is None:
            freq_hz = nw.f
        elif len(freq_hz) != len(nw.f) or not np.allclose(freq_hz, nw.f):
            raise ValueError(f"Frequency grid mismatch: {path}")
        targets.append(nw.s)
    return np.stack(targets, axis=0), freq_hz


def load_hfss_equivalent_simulation(dut_df: pd.DataFrame, settings: dict[str, object]):
    target_s, freq_hz = load_targets_and_freq(dut_df)
    backend = HfssEquivalentDeviceBackend()
    base_rows = []
    for sample_idx, row in dut_df.iterrows():
        blocks = []
        for device_name in base.DEVICE_SEQUENCE:
            s = backend.device_s(row, device_name, freq_hz)
            blocks.append(base.s2abcd(s))
        base_rows.append(np.stack(blocks, axis=0))
        print(f"HFSS-equivalent single-device cascade {sample_idx + 1}/{len(dut_df)}", flush=True)
    return base.SimulationBundle(
        base_abcds=np.stack(base_rows, axis=0),
        target_s=target_s,
        freq_hz=freq_hz,
        simulator_report={
            "backend": "hfss_equivalent_circuit",
            "rdl_checkpoint": str(RDL_CONNECTION2_CHECKPOINT),
            "rdl_model_devices": ["TMRDL", "BSMRDL"],
            "rdl_model_note": "TMRDL and BSMRDL use the new generic LHS400_Connection2 RDL equivalent-circuit checkpoint with their own l/w/h feature mapping.",
            "tsv_checkpoint": str(TSV_CONNECTION2_CHECKPOINT),
            "single_device_source": "HFSS_sim/LHS400_Connection2/train",
            "settings": settings,
        },
    )


def patch_validation_archive() -> None:
    archive = OUTPUT_DIR / "validation_archive.md"
    if not archive.exists():
        return
    text = archive.read_text(encoding="utf-8")
    text = text.replace("# v11 ADS V08-Circuit Shared-to-Multihead Validation", "# v12 HFSS V08-Circuit Symmetric-Multihead Validation")
    text = text.replace("- Entry: `train_v11_shared7_to_multihead12.py`", "- Entry: `train_v12_hfss_v08_symmetric_multihead.py`")
    text = text.replace("ADS single-device S-parameters are cascaded.", "HFSS-derived equivalent-circuit single-device S-parameters are cascaded.")
    text = text.replace(
        "seven parameter networks, each expanded to 12 `30->20->1` connection heads.",
        "seven parameter networks, each expanded to six learned `30->20->1` heads mirrored to 12 connection positions.",
    )
    text += "\n\n## v12 Notes\n\n"
    text += "- Single-device backend: new HFSS-derived equivalent-circuit checkpoints from LHS400_Connection2 data for both RDL and TSV.\n"
    text += "- Symmetric heads: `1,2,3,4,5,6,6,5,4,3,2,1`.\n"
    text += "- Direct-run entry: `code/train_v12_hfss_v08_symmetric_multihead.py`.\n"
    archive.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = OUTPUT_DIR
    base.SIMULATION_BACKEND = "hfss_equivalent_circuit"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.OPT_MAX_NFEV = shared.OPT_MAX_NFEV
    base.collect_samples = lambda: shared.collect_v11_samples(base)
    base.load_single_device_simulation = load_hfss_equivalent_simulation

    shared.RUN_LABEL = RUN_LABEL
    shared.OPTIMIZED_V08_NMSE_FILTER_THRESHOLD = 0.30
    shared.MultiHeadV08ConnectionNet = SymmetricV08ConnectionNet

    settings = {
        "version": "v12",
        "single_device_backend": "hfss_equivalent_circuit",
        "full_chain_dataset": "LHS150_50_Connection2",
        "connection_circuit": "v08_appendix1_7_parameter_pi",
        "rdl_checkpoint": str(RDL_CONNECTION2_CHECKPOINT),
        "tsv_checkpoint": str(TSV_CONNECTION2_CHECKPOINT),
        "tsv_feature_mapping": "TSV Connection2 model input uses r_tsv directly, not 2*r_tsv.",
        "multihead_symmetry": "six learned heads mirrored to twelve positions",
    }
    try:
        summary = shared.run_once(base, settings)
        report_path = OUTPUT_DIR / "training_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["run_label"] = RUN_LABEL
            report["version"] = "v12_hfss_v08_multihead_chain"
            report["workflow"][0] = "HFSS-derived equivalent-circuit single-device S-parameters are cascaded."
            report["workflow"][-1] = "The 20-node layer is expanded into six mirrored connection heads per parameter and fine-tuned with S11/S21 magnitude and wrapped phase loss."
            report["symmetric_heads"] = "1,2,3,4,5,6,6,5,4,3,2,1"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        patch_validation_archive()
        print(summary.to_string(index=False), flush=True)
    except FileNotFoundError as exc:
        readiness = pd.DataFrame(
            [
                {"item": "RDL Connection2 equivalent-circuit checkpoint", "path": str(RDL_CONNECTION2_CHECKPOINT), "exists": RDL_CONNECTION2_CHECKPOINT.exists()},
                {"item": "TSV Connection2 equivalent-circuit checkpoint", "path": str(TSV_CONNECTION2_CHECKPOINT), "exists": TSV_CONNECTION2_CHECKPOINT.exists()},
                {"item": "Full-chain train/test data", "path": str(PROJECT_ROOT / "HFSS_sim" / "LHS150_50_Connection2"), "exists": (PROJECT_ROOT / "HFSS_sim" / "LHS150_50_Connection2").exists()},
            ]
        )
        readiness.to_csv(OUTPUT_DIR / "data_readiness_summary.csv", index=False, encoding="utf-8-sig")
        (OUTPUT_DIR / "validation_archive.md").write_text(
            "\n".join(
                [
                    "# v12 HFSS V08-Circuit Symmetric-Multihead Validation",
                    "",
                    f"- Entry: `{Path(__file__).name}`",
                    "- Status: blocked before training",
                    f"- Reason: {exc}",
                    f"- Output: `{OUTPUT_DIR}`",
                    "",
                    "## Data Readiness",
                    "",
                    base.dataframe_to_markdown(readiness),
                ]
            ),
            encoding="utf-8",
        )
        print(f"[blocked] {exc}", flush=True)
        print(f"[blocked] Validation archive: {OUTPUT_DIR / 'validation_archive.md'}", flush=True)


if __name__ == "__main__":
    main()
