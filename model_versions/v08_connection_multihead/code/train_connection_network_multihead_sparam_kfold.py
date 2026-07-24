# -*- coding: utf-8 -*-
"""Run K-fold validation for the multi-head connection-network S-parameter model.

Run this file directly in VS Code. It reuses
``train_connection_network_multihead_sparam.py`` for model definition,
abnormal HFSS filtering, refined single-device models, training, and plotting.

For each fold:
- one fold is held out as test;
- the next fold is used as validation;
- the remaining folds are used for training.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V08_CODE_DIR = PROJECT_ROOT / "model_versions" / "v08_connection_multihead" / "code"
V07_CODE_DIR = PROJECT_ROOT / "model_versions" / "v07_connection_param_and_sparam_finetune" / "code"
for path in [V08_CODE_DIR, V07_CODE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_connection_network_multihead_sparam as base
import train_connection_network_params as param_train


K_FOLDS = 5
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v08_connection_multihead" / "results" / "connection_network_multihead_sparam_with_cn3_kfold"
RUN_PLOTS_PER_FOLD = True


def make_fold_masks(dut_indices, fold_index, n_folds):
    rng = np.random.default_rng(base.RANDOM_SEED)
    shuffled = np.asarray(dut_indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    fold_ids = np.array_split(shuffled, n_folds)

    test_ids = set(fold_ids[fold_index].tolist())
    val_ids = set(fold_ids[(fold_index + 1) % n_folds].tolist())
    train_ids = set()
    for i, ids in enumerate(fold_ids):
        if i not in {fold_index, (fold_index + 1) % n_folds}:
            train_ids.update(ids.tolist())

    train_mask = np.asarray([int(idx) in train_ids for idx in dut_indices], dtype=bool)
    val_mask = np.asarray([int(idx) in val_ids for idx in dut_indices], dtype=bool)
    test_mask = np.asarray([int(idx) in test_ids for idx in dut_indices], dtype=bool)
    return train_mask, val_mask, test_mask


def rebuild_fold_arrays(base_arrays, fold_index):
    (
        dut_df,
        _x_norm,
        _y_norm,
        y_raw,
        _train_mask,
        _val_mask,
        _test_mask,
        metadata,
        base_abcds,
        target_s,
        dut_indices,
        bad_s21_df,
        sample_weights,
    ) = base_arrays

    train_mask, val_mask, test_mask = make_fold_masks(dut_indices, fold_index, K_FOLDS)
    x_raw = dut_df[param_train.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = base.normalize_train(x_raw, train_mask)

    if base.PHYSICAL_POSITIVE_OUTPUT:
        y_for_model, y_lower, y_upper = base.logit_from_physical_targets(y_raw)
    else:
        y_for_model = y_raw
        y_lower, y_upper = base.physical_bounds(y_raw.shape[1])
    y_norm, y_mean, y_std = base.normalize_train(y_for_model, train_mask)

    fold_metadata = copy.deepcopy(metadata)
    fold_metadata.update(
        {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
            "physical_lower": y_lower.tolist(),
            "physical_upper": y_upper.tolist(),
            "k_folds": K_FOLDS,
            "fold_index": fold_index + 1,
            "fold_train_duts": dut_indices[train_mask].astype(int).tolist(),
            "fold_val_duts": dut_indices[val_mask].astype(int).tolist(),
            "fold_test_duts": dut_indices[test_mask].astype(int).tolist(),
        }
    )

    return (
        dut_df,
        x_norm,
        y_norm,
        y_raw,
        train_mask,
        val_mask,
        test_mask,
        fold_metadata,
        base_abcds,
        target_s,
        dut_indices,
        bad_s21_df,
        sample_weights,
    )


def summarize_split(metrics_df, split_name):
    split_df = metrics_df[metrics_df["split"] == split_name]
    values = split_df["multihead_mse_vs_hfss"]
    return {
        f"{split_name}_count": int(len(split_df)),
        f"{split_name}_mean": float(values.mean()),
        f"{split_name}_median": float(values.median()),
        f"{split_name}_p95": float(values.quantile(0.95)),
        f"{split_name}_p99": float(values.quantile(0.99)),
        f"{split_name}_max": float(values.max()),
        f"{split_name}_direct_mean": float(split_df["direct_mse_vs_hfss"].mean()),
        f"{split_name}_optimized_mean": float(split_df["optimized_mse_vs_hfss"].mean()),
    }


def run_one_fold(base_arrays, fold_index, device):
    fold_dir = OUTPUT_DIR / f"fold_{fold_index + 1:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_DIR = fold_dir
    base.PLOT_SPLIT = "test"
    base.PLOT_DUT_LIMIT = 10

    base.set_seed(base.RANDOM_SEED + fold_index)
    arrays = rebuild_fold_arrays(base_arrays, fold_index)
    dut_df, x_norm, _, _, train_mask, val_mask, test_mask, metadata, _, _, _, bad_s21_df, _ = arrays

    print(
        f"\n===== Fold {fold_index + 1}/{K_FOLDS}: "
        f"train={int(train_mask.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())} =====",
        flush=True,
    )

    model = base.MultiHeadConnectionNet(
        input_dim=x_norm.shape[1],
        connection_count=param_train.CONNECTION_COUNT,
        head_dim=len(param_train.SCALE_COLUMNS),
    ).to(dtype=base.REAL_DTYPE, device=device)

    param_history = base.train_param_pretrain(model, arrays, device) if base.RUN_PARAM_PRETRAIN else pd.DataFrame()
    sparam_history = base.train_sparam(model, arrays, device)
    history_df = pd.concat([param_history, sparam_history], ignore_index=True)
    history_df.to_csv(fold_dir / "multihead_training_history.csv", index=False, encoding="utf-8-sig")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "model_type": "multihead_connection_net_kfold",
            "fold_index": fold_index + 1,
            "k_folds": K_FOLDS,
        },
        fold_dir / "connection_param_multihead_net.pt",
    )

    if not bad_s21_df.empty:
        bad_s21_df.to_csv(fold_dir / "excluded_bad_hfss_s21_samples.csv", index=False, encoding="utf-8-sig")

    metrics_df, pred_df = base.predict_metrics_and_params(model, arrays, device)
    metrics_df.to_csv(fold_dir / "multihead_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(fold_dir / "multihead_param_predictions.csv", index=False, encoding="utf-8-sig")
    plot_dir, n_plots = base.save_plots(metrics_df, pred_df, arrays) if RUN_PLOTS_PER_FOLD else (None, 0)

    summary = {
        "fold": fold_index + 1,
        "output_dir": str(fold_dir),
        "plot_dir": str(plot_dir) if plot_dir is not None else None,
        "n_plots": int(n_plots),
        "n_dut": int(len(dut_df)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "param_pretrain_epochs": int(param_history["epoch"].iloc[-1]) if len(param_history) else 0,
        "sparam_epochs": int(sparam_history["epoch"].iloc[-1]) if len(sparam_history) else 0,
        "best_val_s_loss": float(sparam_history["val_s_loss"].min()) if len(sparam_history) else None,
    }
    summary.update(summarize_split(metrics_df, "test"))
    with open(fold_dir / "fold_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"Fold {fold_index + 1} test mean={summary['test_mean']:.6e}, "
        f"p95={summary['test_p95']:.6e}, max={summary['test_max']:.6e}",
        flush=True,
    )
    return summary, metrics_df[metrics_df["split"] == "test"].copy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)
    print(f"K-fold 输出目录: {OUTPUT_DIR}", flush=True)
    print("预计算一次过滤后的级联数据...", flush=True)
    base_arrays = base.build_training_arrays()

    summaries = []
    test_parts = []
    for fold_index in range(K_FOLDS):
        summary, test_df = run_one_fold(base_arrays, fold_index, device)
        summaries.append(summary)
        test_df["fold"] = fold_index + 1
        test_parts.append(test_df)

    summary_df = pd.DataFrame(summaries)
    all_test_df = pd.concat(test_parts, ignore_index=True)
    summary_df.to_csv(OUTPUT_DIR / "kfold_summary.csv", index=False, encoding="utf-8-sig")
    all_test_df.to_csv(OUTPUT_DIR / "kfold_all_test_metrics.csv", index=False, encoding="utf-8-sig")

    aggregate = {
        "k_folds": K_FOLDS,
        "output_dir": str(OUTPUT_DIR),
        "test_mean_mean": float(summary_df["test_mean"].mean()),
        "test_mean_std": float(summary_df["test_mean"].std(ddof=1)),
        "all_test_mean": float(all_test_df["multihead_mse_vs_hfss"].mean()),
        "all_test_median": float(all_test_df["multihead_mse_vs_hfss"].median()),
        "all_test_p95": float(all_test_df["multihead_mse_vs_hfss"].quantile(0.95)),
        "all_test_p99": float(all_test_df["multihead_mse_vs_hfss"].quantile(0.99)),
        "all_test_max": float(all_test_df["multihead_mse_vs_hfss"].max()),
        "all_test_direct_mean": float(all_test_df["direct_mse_vs_hfss"].mean()),
        "all_test_optimized_mean": float(all_test_df["optimized_mse_vs_hfss"].mean()),
    }
    with open(OUTPUT_DIR / "kfold_report.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print("\nK-fold 验证完成")
    print(summary_df[["fold", "test_mean", "test_median", "test_p95", "test_p99", "test_max"]].to_string(index=False))
    print("\n汇总:")
    for key, value in aggregate.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6e}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
