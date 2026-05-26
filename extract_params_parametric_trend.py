import importlib.util
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


BASE_DIR = os.path.dirname(__file__)
SNP_DIR = os.path.join(BASE_DIR, "RDL_Bottom_Snp")
OUTPUT_CSV = os.path.join(BASE_DIR, "RDL_Bottom_TD_parametric_trend.csv")
SOURCE_SCRIPT = os.path.join(BASE_DIR, "提参2.py")
INIT_CSV = os.path.join(BASE_DIR, "RDL_Bottom_TD_2.csv")

N_DUT = 300
RIDGE_LAMBDA = 0.02
MAX_NFEV = 120


def load_source_module():
    spec = importlib.util.spec_from_file_location("param_source", SOURCE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_header_vars(path):
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


def make_features(x_norm):
    rows = []
    for x in x_norm:
        feat = [1.0]
        feat.extend(x.tolist())
        for i in range(len(x)):
            for j in range(i, len(x)):
                feat.append(x[i] * x[j])
        rows.append(feat)
    return np.asarray(rows)


def rl_model(params, freq):
    R1, R2, R3, L1_nh, L2_nh, L3_nh = params
    L1, L2, L3 = L1_nh * 1e-9, L2_nh * 1e-9, L3_nh * 1e-9
    omega = 2 * np.pi * freq
    R_fit = (
        (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2)
        / ((R1 + R2) ** 2 + omega**2 * L2**2)
        + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    )
    L_fit = (
        (R1**2 * L2) / ((R1 + R2) ** 2 + omega**2 * L2**2)
        + L3 * R3**2 / (R3**2 + omega**2 * L3**2)
        + L1
    )
    return R_fit, L_fit


def fit_gc(G, C, freq):
    Cdc, Ghf, Chf = C[0], G[-1], C[-1]
    C1 = abs(Cdc)
    C2 = abs(Cdc * Chf / max(abs(Cdc - Chf), 1e-30))
    Rsi = abs(C1**2 / max(abs(Ghf * (C1 + C2) ** 2), 1e-30))

    def residual(x):
        Cox, Csi, Rsi_ = x[0] * 1e-12, x[1] * 1e-12, x[2]
        omega = 2 * np.pi * freq
        G_fit = omega**2 * Rsi_ * Cox**2 / (1 + omega**2 * Rsi_**2 * (Cox + Csi) ** 2)
        C_fit = (
            Cox
            + omega**2 * Csi * Rsi_**2 * Cox * (Cox + Csi)
            / (1 + omega**2 * Rsi_**2 * (Cox + Csi) ** 2)
        )
        return np.r_[(G_fit - G) / max(abs(G[-1]), 1e-30), (C_fit - C) / max(abs(C[-1]), 1e-30)]

    x0 = np.array([C1 * 1e12, C2 * 1e12, Rsi])
    lower = np.maximum([x0[0] * 0.2, x0[1] * 0.2, x0[2] * 0.2], [1e-6, 1e-6, 1e-9])
    upper = np.maximum([x0[0] * 5, x0[1] * 5, x0[2] * 5], lower * 1.01)
    res = least_squares(residual, np.clip(x0, lower, upper), bounds=(lower, upper))
    return res.x


def independent_init_params(mod, sample):
    R, L, G, C, freq = sample["R"], sample["L"], sample["G"], sample["C"], sample["freq"]
    *_, params, _ = mod.RLGC_SPICE_rlgc_way3(R, L, G, C, sample["length"], freq, p1=0, p2=len(freq) - 1)
    return params[:6]


def main():
    mod = load_source_module()
    samples = []
    geometry_rows = []
    for i in range(N_DUT):
        path = os.path.join(SNP_DIR, f"dut{i}.s2p")
        if not os.path.exists(path):
            print(f"missing {path}, skip")
            continue
        variables = parse_header_vars(path)
        length = variables["ldown"] * 1e-6
        S11, S12, S21, S22, _, _, _, _, freq = mod.path_S2P(path)
        A, B, Cmat, D = mod.S_ABCD(S11, S12, S21, S22)
        R, L, G, C, _, _ = mod.ABCD_RLGC(A, B, Cmat, D, freq, length)
        geom = [variables["ldown"], variables["wdown"], variables["tdown"], variables["htsv"], variables["p1"]]
        geometry_rows.append(geom)
        samples.append({"dut": i, "vars": variables, "length": length, "R": R, "L": L, "G": G, "C": C, "freq": freq})

    geometry = np.asarray(geometry_rows, dtype=float)
    geom_mean = geometry.mean(axis=0)
    geom_std = np.maximum(geometry.std(axis=0), 1e-12)
    phi = make_features((geometry - geom_mean) / geom_std)

    if os.path.exists(INIT_CSV):
        init_df = pd.read_csv(INIT_CSV)
        init_params = init_df[["R1", "R2", "R3", "L1", "L2", "L3"]].to_numpy(dtype=float)[: len(samples)]
    else:
        init_params = np.asarray([independent_init_params(mod, s) for s in samples])

    init_params = np.clip(init_params, [50, 100, 50, 100, 1, 1], [8000, 20000, 8000, 3000, 8000, 300])
    y0 = np.log(init_params)
    coef0 = np.linalg.lstsq(phi, y0, rcond=None)[0]
    coef0_flat = coef0.ravel()

    lower_param = np.log(np.array([50, 100, 50, 100, 1, 1], dtype=float))
    upper_param = np.log(np.array([8000, 20000, 8000, 3000, 8000, 300], dtype=float))

    def unpack(coef_flat):
        coef = coef_flat.reshape(phi.shape[1], 6)
        log_params = phi @ coef
        log_params = np.clip(log_params, lower_param, upper_param)
        return np.exp(log_params)

    def residual_all(coef_flat):
        params_all = unpack(coef_flat)
        residuals = []
        for params, sample in zip(params_all, samples):
            R_fit, L_fit = rl_model(params, sample["freq"])
            residuals.append((R_fit - sample["R"]) / max(abs(sample["R"][-1]), 1e-30))
            residuals.append((L_fit - sample["L"]) / max(abs(sample["L"][-1]), 1e-30))
        residuals.append(np.sqrt(RIDGE_LAMBDA) * (coef_flat - coef0_flat))
        return np.concatenate(residuals)

    print(f"optimizing {coef0_flat.size} coefficients for {len(samples)} samples")
    res = least_squares(
        residual_all,
        coef0_flat,
        max_nfev=MAX_NFEV,
        verbose=2,
        xtol=1e-8,
        ftol=1e-8,
    )

    params_all = unpack(res.x)
    rows = []
    for params, sample in zip(params_all, samples):
        R_fit, L_fit = rl_model(params, sample["freq"])
        rmse = float(
            np.sqrt(
                np.mean(
                    np.r_[
                        (R_fit - sample["R"]) / max(abs(sample["R"][-1]), 1e-30),
                        (L_fit - sample["L"]) / max(abs(sample["L"][-1]), 1e-30),
                    ]
                    ** 2
                )
            )
        )
        Cox, Csi, Rsi = fit_gc(sample["G"], sample["C"], sample["freq"])
        v = sample["vars"]
        rows.append([v["ldown"], v["wdown"], v["tdown"], v["htsv"], v["p1"], *params, Cox, Csi, Rsi, rmse])

    headers = ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl", "R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi", "rmse"]
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
