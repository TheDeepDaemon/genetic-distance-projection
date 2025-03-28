from gdp import GenomeData, join_genomes, join_genomes_list, GenomeDataCollector
from local_util import load_program_arguments, load_data, get_numbered_unique_fpath, get_subset
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

    print(f"Pre-loading from: {data_source_path}")

    # data_storage run type
    run_type = args["run_type"]

    # where to store the genome data_storage
    save_fpath = f"{pre_load_data_dir}/{run_type}_genome_data.zip"
    save_fpath = get_numbered_unique_fpath(save_fpath)

    data_collector = GenomeDataCollector()
    data_collector.load(f"{os.path.join(data_source_path, run_type)}.zip")

    use_node_gene_data = args["use_node_gene_data"]
    use_edge_gene_data = args["use_edge_gene_data"]
    use_edge_weights_data = args["use_edge_weights_data"]
    use_recurrent_edge_gene_data = args["use_recurrent_edge_gene_data"]
    use_recurrent_edge_weights_data = args["use_recurrent_edge_weights_data"]

    if use_node_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: [f"node_id:{gn["n"]}" for gn in data_entry],
            key="nodes")

    if use_edge_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: [f"edge_id:{ge["n"]}" for ge in data_entry],
            key="edges")

    if use_edge_weights_data:
        data_collector.convert_info_to_gene_values(
            func=lambda data_entry: {f"edge_weight:{ge["n"]}": float(ge["weight"]) for ge in data_entry},
            key="edges")

    if use_recurrent_edge_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: [f"recurrent_edge:{gre["n"]}" for gre in data_entry],
            key="recurrent_edges")

    if use_recurrent_edge_weights_data:
        data_collector.convert_info_to_gene_values(
            func=lambda data_entry: {f"recurrent_edge_weight:{gre["n"]}": float(gre["weight"]) for gre in data_entry},
            key="recurrent_edges")

    genome_data = GenomeData()

    genome_data.init_data(genome_data_collector=data_collector)

    # set genome IDs
    genome_ids = data_collector.get_unique_genome_id_list()

    genome_parents = data_collector.get_genome_attribute_by_key("parents")

    # set parent-child relations
    relations = []
    for genome_id, parnts in genome_parents.items():
        for parent in parnts:
            relations.append((parent, genome_id))

    # init graph data_storage with genome IDs and parent-child relations
    genome_data.init_graph_data(genome_ids=genome_ids, relations=relations)

    # set genome fitnesses
    fitnesses = data_collector.get_genome_attribute_by_key("fitness")

    genome_data.set_genome_fitnesses(fitnesses=fitnesses)

    # make sure this directory exists, so we can output to it
    os.makedirs(pre_load_data_dir, exist_ok=True)

    identifying_keys = [
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
