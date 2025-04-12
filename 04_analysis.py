from gdp import GenomeMatrix, GenomeDataCollector
from gdp.program_arguments import ProgramArguments
from analysis import plot_genes_vs_generation_number, get_all_distances
from enum import Enum
import os


class AnalysisMode(Enum):
    COMPARE_DIST = 1
    PLOT_UNIQUE_GENES = 2


def plot_genes_vs_generation_number_main(program_args: ProgramArguments):

    data_collector = GenomeDataCollector()
    data_collector.load(f"data/{program_args.run_type}.zip")

    plot_genes_vs_generation_number(program_args, data_collector)


def compare_scaled_distances_main(program_args: ProgramArguments):

    identifying_keys = [
        "reduction_type",
        "run_type",
        "use_node_gene_data",
        "use_edge_gene_data",
        "use_edge_weights_data",
        "use_recurrent_edge_gene_data",
        "use_recurrent_edge_weights_data"
    ]

    # find the genome data to use
    load_fname = GenomeMatrix.find_latest_genome_data(
        data_dir="reduced_genome_data",
        identifying_args=program_args.get_subset(identifying_keys))

    # create the genome data storage class to be used for visuals
    genome_matrix = GenomeMatrix()

    # load the processed data storage from the directory
    genome_matrix.load_data(
        zip_fpath=os.path.join("reduced_genome_data", load_fname),
        identifying_args=program_args.get_subset(identifying_keys))

    dist = get_all_distances(
        genome_matrix=genome_matrix.genome_data_mat,
        positions_matrix=genome_matrix.position_data)

    print(f"Distances: {dist}")


def main(analysis_mode, program_args):
    if analysis_mode == AnalysisMode.COMPARE_DIST:
        compare_scaled_distances_main(program_args)
    elif analysis_mode == AnalysisMode.PLOT_UNIQUE_GENES:
        plot_genes_vs_generation_number_main(program_args)
    else:
        raise ValueError("Analysis mode selected is invalid.")


if __name__=="__main__":
    analysis_mode = AnalysisMode.COMPARE_DIST
    program_args = ProgramArguments()
    main(analysis_mode, program_args)
