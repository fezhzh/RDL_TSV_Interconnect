# -*- coding: utf-8 -*-
"""Shared plotting helpers for RDL/TSV scripts."""

import os
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODEL_COLORS = {
    "mat2": "#dc2626",
    "mat3": "#059669",
    "mat4": "#7e22ce",
    "new_s_finetuned": "#ea580c",
    "new": "#ea580c",
}


def db20(value):
    return 20.0 * np.log10(np.maximum(np.abs(value), 1e-300))


def configure_comparison_matplotlib() -> None:
    """Apply a consistent style for model comparison figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f8fb",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1f2937",
            "axes.labelcolor": "#1f2937",
            "axes.titlecolor": "#111827",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cbd5e1",
            "savefig.facecolor": "#f6f8fb",
            "savefig.bbox": "tight",
        }
    )


def style_frequency_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, pad=8)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)


def save_model_case_plot(out_path, nw_hfss, pred_by_model: Mapping[str, np.ndarray], title: str) -> None:
    """Save a 2x2 S-parameter magnitude comparison for one case."""
    configure_comparison_matplotlib()
    freq_ghz = nw_hfss.f / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=16, fontweight="semibold")

    summary_parts = []
    for model_name, pred_s in pred_by_model.items():
        mse = np.mean(np.abs(pred_s - nw_hfss.s) ** 2)
        s21_mae = np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(nw_hfss.s[:, 1, 0])))
        summary_parts.append(f"{model_name}: MSE {mse:.3e}, S21 MAE {s21_mae:.3f} dB")
    if summary_parts:
        fig.text(0.02, 0.953, "    ".join(summary_parts), ha="left", va="top", fontsize=10, color="#475569")

    ports = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]
    for ax, (m, n, name) in zip(axes.ravel(), ports):
        ax.plot(freq_ghz, db20(nw_hfss.s[:, m, n]), label="HFSS", color="#1f77b4", linewidth=1.8)
        for model_name, pred_s in pred_by_model.items():
            ax.plot(
                freq_ghz,
                db20(pred_s[:, m, n]),
                label=model_name,
                color=DEFAULT_MODEL_COLORS.get(model_name, "#7e22ce"),
                linestyle="--",
                linewidth=1.8,
            )
        style_frequency_axis(ax, f"{name} magnitude", "Magnitude (dB)")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.91, wspace=0.18, hspace=0.28)
    fig.savefig(out_path)
    plt.close(fig)


def save_model_summary_plots(out_dir, summary_df, model_names: Sequence[str]) -> None:
    """Save dataset-level model error trend plots."""
    configure_comparison_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.suptitle("RDL_Bottom Model Error Summary", x=0.02, y=0.98, ha="left", fontsize=16, fontweight="semibold")
    fig.text(
        0.02,
        0.925,
        f"Valid cases: {len(summary_df)}    Lower values indicate closer agreement with HFSS",
        ha="left",
        fontsize=10,
        color="#475569",
    )
    x = np.arange(len(summary_df))

    for model in model_names:
        color = DEFAULT_MODEL_COLORS.get(model, "#1f77b4")
        axes[0].plot(
            x,
            summary_df[f"{model}_vs_hfss_complex_mse"].to_numpy(dtype=float),
            label=model,
            color=color,
            linewidth=1.5,
        )
        axes[1].plot(
            x,
            summary_df[f"{model}_vs_hfss_s21_db_mae"].to_numpy(dtype=float),
            label=model,
            color=color,
            linewidth=1.5,
        )

    axes[0].set_yscale("log")
    axes[0].set_title("Complex S MSE vs HFSS", loc="left", fontsize=12, pad=8)
    axes[0].set_xlabel("Case index")
    axes[0].set_ylabel("MSE")
    axes[1].set_title("S21 magnitude MAE vs HFSS", loc="left", fontsize=12, pad=8)
    axes[1].set_xlabel("Case index")
    axes[1].set_ylabel("MAE (dB)")

    for ax in axes:
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.84, wspace=0.18)
    fig.savefig(Path(out_dir) / "summary_error_trends.png")
    plt.close(fig)


def plot_2ports_Leq(Yparameters, names, freqs):
    fig, axes = plt.subplots(1, 2)
    for Yp, freq, name in zip(Yparameters, freqs, names):
        Leq = np.imag(1 / Yp[0, 0]) / 2 / np.pi / freq
        Q = np.imag(1 / Yp[0, 0]) / np.real(1 / Yp[0, 0])
        axes[0].plot(freq, Leq, label=f"{name}")
        axes[1].plot(freq, Q, label=f"{name}")

    axes[0].set_title("Equivalent Inductance", fontsize=12)
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].legend()
    axes[1].set_title("Quality Factor", fontsize=12)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].legend()
    plt.tight_layout()


def plot_2ports_Req(Yparameters, names, freqs):
    fig, axes = plt.subplots(1, 2)
    for Yp, freq, name in zip(Yparameters, freqs, names):
        Req = np.real(1 / Yp[0, 0])
        Q = np.imag(1 / Yp[0, 0]) / np.real(1 / Yp[0, 0])
        axes[0].plot(freq, Req, label=f"{name}")
        axes[1].plot(freq, Q, label=f"{name}")

    axes[0].set_title("Equivalent Resistance", fontsize=12)
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].legend()
    axes[1].set_title("Quality Factor", fontsize=12)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].legend()
    plt.tight_layout()


def plot_2ports_S(Sparameters, names, type, freqs):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    if type == "RI":
        titles = [
            "S11 Real",
            "S11 Imag",
            "S12 Real",
            "S12 Imag",
            "S21 Real",
            "S21 Imag",
            "S22 Real",
            "S22 Imag",
        ]
        ylabels = ["Real"] * 8
        extractors = [
            lambda Sp: Sp[0, 0].real,
            lambda Sp: Sp[0, 0].imag,
            lambda Sp: Sp[0, 1].real,
            lambda Sp: Sp[0, 1].imag,
            lambda Sp: Sp[1, 0].real,
            lambda Sp: Sp[1, 0].imag,
            lambda Sp: Sp[1, 1].real,
            lambda Sp: Sp[1, 1].imag,
        ]
    elif type == "MP":
        titles = [
            "S11 |dB|",
            "S11 Phase",
            "S12 |dB|",
            "S12 Phase",
            "S21 |dB|",
            "S21 Phase",
            "S22 |dB|",
            "S22 Phase",
        ]
        ylabels = ["dB", "deg"] * 4
        extractors = [
            lambda Sp: db20(Sp[0, 0]),
            lambda Sp: np.angle(Sp[0, 0], deg=True),
            lambda Sp: db20(Sp[0, 1]),
            lambda Sp: np.angle(Sp[0, 1], deg=True),
            lambda Sp: db20(Sp[1, 0]),
            lambda Sp: np.angle(Sp[1, 0], deg=True),
            lambda Sp: db20(Sp[1, 1]),
            lambda Sp: np.angle(Sp[1, 1], deg=True),
        ]
    else:
        raise ValueError(f"Unknown type: {type}")

    flat_axes = axes.ravel()
    for Sp, name, freq in zip(Sparameters, names, freqs):
        for ax, extractor in zip(flat_axes, extractors):
            ax.plot(freq, extractor(Sp), label=f"{name}")

    for ax, title, ylabel in zip(flat_axes, titles, ylabels):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.suptitle(f"S-Parameter Plot ({type})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_RLGC(RLGCs, freqs, names):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["R", "L", "G", "C"]
    ylabels = ["Resistance (Ohm)", "Inductance (H)", "Conductance (S)", "Capacitance (F)"]

    for RLGC, freq, name in zip(RLGCs, freqs, names):
        for ax, values, title, ylabel in zip(axes.ravel(), RLGC, titles, ylabels):
            ax.plot(freq, values, label=f"{name}")
            ax.set_title(title)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel(ylabel)
            ax.legend()

    plt.tight_layout()


def save_current_figure(save_path, dpi: int = 300) -> None:
    """Save current pyplot figure, creating the destination directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
