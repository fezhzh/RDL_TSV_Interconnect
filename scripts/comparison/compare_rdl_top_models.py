"""Compare RDL_Top MATLAB .mat models against HFSS s2p files.

This is the RDL_Top entry point for the reusable comparison code in
``compare_rdl_bottom_models.py``.
"""

from compare_rdl_bottom_models import compare_models
import compare_rdl_bottom_models as compare_core


def build_arg_parser():
    parser = compare_core.build_arg_parser()
    parser.description = "Compare RDL_Top MATLAB .mat models against HFSS s2p files."
    parser.set_defaults(
        hfss_dir="snp_data/RDL_Top_Snp",
        case_csv="training_datasets/RDL_Top_TD_4.csv",
        out_dir="model_results/comparison/RDL_Top_model_compare",
        length_param="lrdl",
        device_name="RDL_Top",
        output_prefix="rdl_top",
    )
    parser.set_defaults(progress_every=25)
    return parser


def main():
    compare_core.FEATURE_ORDER = ["lrdl", "wrdl", "trdl", "htsv", "p1"]
    compare_core.CSV_FEATURE_ORDER = ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl"]

    args = build_arg_parser().parse_args()
    model_configs = [
        {"label": "mat1", "type": "matlab", "path": "device_models/RDL_TSV_mat1", "prefix": "RDL_Top_"},
        {"label": "mat2", "type": "matlab", "path": "device_models/RDL_TSV_mat2", "prefix": "RDL_Top_"},
        {"label": "mat3", "type": "matlab", "path": "device_models/RDL_TSV_mat3", "prefix": "RDL_Top_"},
        {"label": "mat4", "type": "matlab", "path": "device_models/RDL_TSV_mat4", "prefix": "RDL_Top_"},
    ]

    compare_models(args, model_configs, reference_model="mat2")


if __name__ == "__main__":
    main()

