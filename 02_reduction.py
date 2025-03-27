from gdp import GenomeData
from local_util import load_program_arguments, get_numbered_unique_fpath, get_subset
import os


def main():

    # data directory names
    pre_load_data_dir = "pre_loaded_genome_data"
    reduced_data_dir = "reduced_genome_data"

    # load the program arguments
    args = load_program_arguments()

    # what kind of dimensionality reduction to do
    reduction_type = args["reduction_type"]

    # data_storage run type
    run_type = args["run_type"]

    identifying_keys = [
        "run_type",
        "use_node_gene_data",
        "use_edge_gene_data",
        "use_edge_weights_data",
        "use_recurrent_edge_gene_data",
        "use_recurrent_edge_weights_data"
    ]

    # find the preload to use
    genome_data_fname = GenomeData.find_latest_genome_data(
        data_dir=pre_load_data_dir, identifying_args=get_subset(args, identifying_keys))

    # where to store the genome data_storage
    os.makedirs(reduced_data_dir, exist_ok=True)
    save_fpath = f"{reduced_data_dir}/{reduction_type}-{run_type}_genome_data.zip"
    save_fpath = get_numbered_unique_fpath(save_fpath)

    # create genome data_storage object
    genome_data = GenomeData()

    # load the pre-loaded data_storage
    genome_data.load_data(
        zip_fpath=os.path.join(pre_load_data_dir, genome_data_fname),
        identifying_args=get_subset(args, identifying_keys))

    # do reduction
    genome_data.reduce_genome(reduction_type=reduction_type, args=args)

    identifying_keys = [
        "reduction_type",
        "run_type",
        "use_node_gene_data",
        "use_edge_gene_data",
        "use_edge_weights_data",
        "use_recurrent_edge_gene_data",
        "use_recurrent_edge_weights_data"
    ]

    # save the data_storage and reduced data_storage
    genome_data.save_data(
        zip_fpath=save_fpath, identifying_args=get_subset(args, identifying_keys))


if __name__=="__main__":
    main()
