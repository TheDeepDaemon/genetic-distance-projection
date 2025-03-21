from gdp import GenomeData, join_genomes
from local_util import load_program_arguments, load_data, get_numbered_unique_fpath
import os
import argparse


def main(data_source_path):

    # data directory name
    pre_load_data_dir = "pre_loaded_genome_data"

    # load the program arguments
    args = load_program_arguments()

    if data_source_path is None:
        # where to get the data_storage from
        data_source_path = args["data_source_path"]

        if data_source_path is None:
            error_message = ("\'data_source_path\' is not set. "
                             "You must set the \'data_source_path\' variable before continuing.")
            raise ValueError(error_message)

    # data_storage run type
    run_type = args["run_type"]

    # where to store the genome data_storage
    save_fpath = f"{pre_load_data_dir}/{run_type}_genome_data.zip"
    save_fpath = get_numbered_unique_fpath(save_fpath)

    # load the source data_storage (from the EXAMM run)
    genome_data_list = list(load_data(data_filepath=f"{os.path.join(data_source_path, run_type)}.json").values())

    node_genes = {
        data_entry["generation_number"]: [gn["n"] for gn in data_entry["nodes"]]
        for data_entry in genome_data_list}

    edge_genes = {
        data_entry["generation_number"]: [ge["n"] for ge in data_entry["edges"]]
        for data_entry in genome_data_list}

    # genome data_storage from node genes
    node_gene_data = GenomeData()
    node_gene_data.init_data(node_genes)

    # genome data_storage from edge genes
    edge_gene_data = GenomeData()
    edge_gene_data.init_data(edge_genes)

    # get combined genome data_storage
    genome_data = join_genomes(node_gene_data, edge_gene_data)

    # set genome IDs
    genome_ids = list({data_entry["generation_number"] for data_entry in genome_data_list})

    # set parent-child relations
    relations = []
    for data_entry in genome_data_list:
        for parent in data_entry["parents"]:
            relations.append((parent, data_entry["generation_number"]))

    # init graph data_storage with genome IDs and parent-child relations
    genome_data.init_graph_data(genome_ids=genome_ids, relations=relations)

    # set genome fitnesses
    fitnesses = {data_entry["generation_number"]: data_entry["fitness"] for data_entry in genome_data_list}

    genome_data.set_genome_fitnesses(fitnesses=fitnesses)

    # make sure this directory exists, so we can output to it
    os.makedirs(pre_load_data_dir, exist_ok=True)

    # save the data_storage and reduced data_storage
    genome_data.save_data(zip_fpath=save_fpath, run_type=run_type)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pre-load the data to streamline further processing.")

    parser.add_argument(
        "data_source_path",
        nargs="?",
        default=None,
        type=str,
        help="Filename of the directory containing EA run data."
    )

    args = parser.parse_args()

    main(args.data_source_path)
