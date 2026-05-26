import importlib.util
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


BASE_DIR = os.path.dirname(__file__)
SNP_DIR = os.path.join(BASE_DIR, "RDL_Bottom_Snp")
OUTPUT_CSV = os.path.join(BASE_DIR, "RDL_Bottom_TD_dp_trend.csv")
SOURCE_SCRIPT = os.path.join(BASE_DIR, "提参2.py")

N_DUT = 300
SMOOTH_LAMBDA = 0.08
RMSE_WEIGHT = 1.0
SORT_COLUMNS = ["ldown", "wdown", "tdown", "htsv", "p1"]


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


def rl_residual(params, R, L, freq):
    R_fit, L_fit = rl_model(params, freq)
    r_scale = max(abs(R[-1]), 1e-30)
    l_scale = max(abs(L[-1]), 1e-30)
    return np.r_[((R_fit - R) / r_scale), ((L_fit - L) / l_scale)]


def heuristic_init(R, L, p2):
    Rdc, Ldc = abs(R[0]), abs(L[0])
    Rhf, Lhf = abs(R[p2]), abs(L[p2])
    R3 = max(abs(R[p2 // 2]), 1e-9)
    L3 = max(abs(Ldc - abs(L[p2 // 2])), 1e-12)
    R1 = max(Rhf - R3, Rdc * 1.2, 1e-9)
    denom = max(1.0 / Rdc - 1.0 / R1, 1e-12)
    R2 = max(1.0 / denom, 1e-9)
    L1 = max(Lhf, 1e-12)
    L2 = max(abs((Ldc - Lhf - L3) * (R1 + R2) ** 2 / R1**2), 1e-12)
    return np.array([R1, R2, R3, L1 * 1e9, L2 * 1e9, L3 * 1e9])


def fit_rl_candidates(R, L, freq):
    p2 = len(freq) - 1
    base = heuristic_init(R, L, p2)
    lower = np.array([50.0, 100.0, 50.0, 100.0, 1.0, 1.0])
    upper = np.array([8000.0, 20000.0, 8000.0, 3000.0, 8000.0, 300.0])
    seed_scales = [
        [1, 1, 1, 1, 1, 1],
        [0.7, 2.5, 1.3, 1, 2, 1],
        [1.3, 0.6, 0.8, 1, 0.4, 1.4],
        [0.9, 5.0, 1.8, 1, 4.0, 0.8],
        [1.8, 0.4, 0.7, 1, 0.3, 1.8],
        [1.0, 1.0, 2.0, 1, 1.0, 0.6],
        [0.6, 4.0, 2.0, 1, 3.0, 0.7],
    ]
    candidates = []
    seen = set()
    for scale in seed_scales:
        x0 = np.clip(base * np.array(scale), lower, upper)
        res = least_squares(
            rl_residual,
            x0,
            bounds=(lower, upper),
            args=(R, L, freq),
            max_nfev=800,
            xtol=1e-10,
            ftol=1e-10,
        )
        params = res.x
        key = tuple(np.round(np.log10(params), 3))
        if key in seen:
            continue
        seen.add(key)
        rmse = float(np.sqrt(np.mean(rl_residual(params, R, L, freq) ** 2)))
        candidates.append({"params": params, "rmse": rmse})
    candidates.sort(key=lambda item: item["rmse"])
    return candidates[:6]


def fit_gc(G, C, freq):
    Gdc, Cdc = G[0], C[0]
    Ghf, Chf = G[-1], C[-1]
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
    Cox, Csi, Rsi = res.x
    return Cox, Csi, Rsi


def transition_cost(prev_params, next_params):
    prev_log = np.log(np.maximum(prev_params, 1e-30))
    next_log = np.log(np.maximum(next_params, 1e-30))
    weights = np.array([1.0, 0.35, 1.0, 0.6, 0.25, 0.6])
    diff = (next_log - prev_log) * weights
    return float(np.mean(diff**2))


def choose_smooth_path(candidate_lists):
    n = len(candidate_lists)
    costs = [np.full(len(cands), np.inf) for cands in candidate_lists]
    back = [np.full(len(cands), -1, dtype=int) for cands in candidate_lists]
    costs[0] = np.array([RMSE_WEIGHT * c["rmse"] for c in candidate_lists[0]])

    for i in range(1, n):
        for k, cand in enumerate(candidate_lists[i]):
            best_cost = np.inf
            best_j = -1
            for j, prev in enumerate(candidate_lists[i - 1]):
                cost = (
                    costs[i - 1][j]
                    + RMSE_WEIGHT * cand["rmse"]
                    + SMOOTH_LAMBDA * transition_cost(prev["params"], cand["params"])
                )
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
            costs[i][k] = best_cost
            back[i][k] = best_j

    idx = int(np.argmin(costs[-1]))
    path = [idx]
    for i in range(n - 1, 0, -1):
        idx = int(back[i][idx])
        path.append(idx)
    return list(reversed(path))


def main():
    mod = load_source_module()
    samples = []
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
        samples.append(
            {
                "dut": i,
                "vars": variables,
                "R": R,
                "L": L,
                "G": G,
                "C": C,
                "freq": freq,
            }
        )

    order = sorted(range(len(samples)), key=lambda idx: tuple(samples[idx]["vars"][c] for c in SORT_COLUMNS))
    candidate_lists = []
    for pos, idx in enumerate(order):
        s = samples[idx]
        cands = fit_rl_candidates(s["R"], s["L"], s["freq"])
        candidate_lists.append(cands)
        print(f"{pos + 1}/{len(order)} dut{s['dut']}: {len(cands)} candidates, best rmse={cands[0]['rmse']:.4g}")

    path = choose_smooth_path(candidate_lists)
    selected = [candidate_lists[i][path[i]] for i in range(len(path))]

    rows_by_original = {}
    for idx_in_order, sample_idx in enumerate(order):
        s = samples[sample_idx]
        v = s["vars"]
        rl_params = selected[idx_in_order]["params"]
        Cox, Csi, Rsi = fit_gc(s["G"], s["C"], s["freq"])
        rows_by_original[s["dut"]] = [
            v["ldown"],
            v["wdown"],
            v["tdown"],
            v["htsv"],
            v["p1"],
            *rl_params,
            Cox,
            Csi,
            Rsi,
            selected[idx_in_order]["rmse"],
        ]

    headers = ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl", "R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi", "rmse"]
    df = pd.DataFrame([rows_by_original[k] for k in sorted(rows_by_original)], columns=headers)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
