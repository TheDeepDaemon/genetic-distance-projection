from gdp import GenomeMatrix
from gdp.program_arguments import ProgramArguments
from local_util import get_numbered_unique_fpath
import os


def main():
    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # LOAD PROGRAM ARGUMENTS

    # load the program arguments
    program_args = ProgramArguments()

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # LOAD GENOME DATA AND REDUCE

    input_identifying_keys = [
        "run_type",
        "use_node_gene_data",
        "use_edge_gene_data",
        "use_edge_weights_data",
        "use_recurrent_edge_gene_data",
        "use_recurrent_edge_weights_data"
    ]

    # find the preload to use
    genome_data_fname = GenomeMatrix.find_latest_genome_data(
        data_dir="pre_loaded_genome_data", identifying_args=program_args.get_subset(input_identifying_keys))

    # create genome data object
    genome_matrix = GenomeMatrix()

    # load the pre-loaded data
    genome_matrix.load_data(
        zip_fpath=os.path.join("pre_loaded_genome_data", genome_data_fname),
        identifying_args=program_args.get_subset(input_identifying_keys))

    # do reduction
    genome_matrix.reduce_genome(reduction_type=program_args.reduction_type, args=program_args.args)

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # SAVE THE REDUCED DATA

    output_identifying_keys = [
        "reduction_type",
        "run_type",
        "use_node_gene_data",
        "use_edge_gene_data",
        "use_edge_weights_data",
        "use_recurrent_edge_gene_data",
        "use_recurrent_edge_weights_data"
    ]

    # where to store the genome data
    os.makedirs("reduced_genome_data", exist_ok=True)
    save_fpath = f"reduced_genome_data/{program_args.reduction_type}-{program_args.run_type}_genome_data.zip"
    save_fpath = get_numbered_unique_fpath(save_fpath)

    # save the data_storage and reduced data_storage
    genome_matrix.save_data(
        zip_fpath=save_fpath, identifying_args=program_args.get_subset(output_identifying_keys))


if __name__=="__main__":
    main()
