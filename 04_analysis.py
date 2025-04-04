from analysis import compare_scaled_distances, plot_genes_vs_generation_number
from enum import Enum


class AnalysisMode(Enum):
    COMPARE_DIST = 1
    PLOT_UNIQUE_GENES = 2


def main(analysis_mode):
    if analysis_mode == AnalysisMode.COMPARE_DIST:
        compare_scaled_distances()
    elif analysis_mode == AnalysisMode.PLOT_UNIQUE_GENES:
        plot_genes_vs_generation_number()
    else:
        raise ValueError("Analysis mode selected is invalid.")


if __name__=="__main__":
    analysis_mode = AnalysisMode.COMPARE_DIST
    main(analysis_mode)
