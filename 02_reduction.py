from gdp import GenomeMatrix
from local_util import load_program_arguments, get_numbered_unique_fpath, get_subset
import os


def main():
    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # LOAD AND UNPACK PROGRAM ARGUMENTS

    # load the program arguments
    args = load_program_arguments()

    # what kind of dimensionality reduction to do
    reduction_type = args["reduction_type"]

    # data_storage run type
    run_type = args["run_type"]

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
        data_dir="pre_loaded_genome_data", identifying_args=get_subset(args, input_identifying_keys))

    # create genome data object
    genome_data = GenomeMatrix()

    # load the pre-loaded data
    genome_data.load_data(
        zip_fpath=os.path.join("pre_loaded_genome_data", genome_data_fname),
        identifying_args=get_subset(args, input_identifying_keys))

    # do reduction
    genome_data.reduce_genome(reduction_type=reduction_type, args=args)

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
    save_fpath = f"reduced_genome_data/{reduction_type}-{run_type}_genome_data.zip"
    save_fpath = get_numbered_unique_fpath(save_fpath)

    # save the data_storage and reduced data_storage
    genome_data.save_data(
        zip_fpath=save_fpath, identifying_args=get_subset(args, output_identifying_keys))


if __name__=="__main__":
    main()
