"""Compare TSV MATLAB .mat models against HFSS s2p files.

This is the TSV entry point for the reusable comparison code in
``compare_rdl_bottom_models.py``.
"""

from types import SimpleNamespace

from compare_rdl_bottom_models import compare_models
import compare_rdl_bottom_models as compare_core


BASE_DIR = None
HFSS_DIR = "snp_data/TSV_Snp"
CASE_CSV = "training_datasets/TSV_TD_4.csv"
OUT_DIR = "model_results/comparison/TSV_model_compare"
LENGTH_PARAM = "htsv"
DEVICE_NAME = "TSV"
OUTPUT_PREFIX = "tsv"

# Set these values in code when running directly from VS Code.
LIMIT = None
NO_PLOTS = False
PLOT_ALL = False
PLOT_FIRST = 3
PROGRESS_EVERY = 25


def build_args():
    return SimpleNamespace(
        base_dir=BASE_DIR,
        hfss_dir=HFSS_DIR,
        case_csv=CASE_CSV,
        out_dir=OUT_DIR,
        length_param=LENGTH_PARAM,
        device_name=DEVICE_NAME,
        output_prefix=OUTPUT_PREFIX,
        limit=LIMIT,
        no_plots=NO_PLOTS,
        plot_all=PLOT_ALL,
        plot_first=PLOT_FIRST,
        progress_every=PROGRESS_EVERY,
    )


def main():
    compare_core.FEATURE_ORDER = ["dtsv", "htsv", "p1"]
    compare_core.CSV_FEATURE_ORDER = ["d_tsv", "h_tsv", "p_rdl"]

    args = build_args()
    model_configs = [
        {"label": "mat2", "type": "matlab", "path": "device_models/RDL_TSV_mat2", "prefix": "TSV_"},
        {"label": "mat4", "type": "matlab", "path": "device_models/RDL_TSV_mat4", "prefix": "TSV_"},
    ]

    compare_models(args, model_configs, reference_model="mat2")


if __name__ == "__main__":
    main()
