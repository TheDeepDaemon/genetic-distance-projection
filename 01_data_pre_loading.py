from gdp import GenomeMatrix, GenomeDataCollector
from local_util import load_program_arguments, get_numbered_unique_fpath, get_subset
import os


def main():
    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # LOAD AND UNPACK PROGRAM ARGUMENTS

    # load the program arguments
    args = load_program_arguments()

    # data_storage run type
    run_type = args["run_type"]

    use_node_gene_data = args["use_node_gene_data"]
    use_edge_gene_data = args["use_edge_gene_data"]
    use_edge_weights_data = args["use_edge_weights_data"]
    use_recurrent_edge_gene_data = args["use_recurrent_edge_gene_data"]
    use_recurrent_edge_weights_data = args["use_recurrent_edge_weights_data"]

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # LOAD AND SET GENOME DATA

    data_collector = GenomeDataCollector()
    data_collector.load(f"data/{run_type}.zip")

    if use_node_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: ["node_id:{}".format(gn["n"]) for gn in data_entry],
            key="nodes")

    if use_edge_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: ["edge_id:{}".format(ge["n"]) for ge in data_entry],
            key="edges")

    if use_edge_weights_data:
        data_collector.convert_info_to_gene_values(
            func=lambda data_entry: {"edge_weight:{}".format(ge["n"]): float(ge["weight"]) for ge in data_entry},
            key="edges")

    if use_recurrent_edge_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: ["recurrent_edge:{}".format(gre["n"]) for gre in data_entry],
            key="recurrent_edges")

    if use_recurrent_edge_weights_data:
        data_collector.convert_info_to_gene_values(
            func=lambda data_entry: {"recurrent_edge_weight:{}".format(gre["n"]): float(gre["weight"]) for gre in data_entry},
            key="recurrent_edges")

    print("Data encoded.")

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # NUMERICALLY ENCODE INTO A MATRIX

    genome_data = GenomeMatrix()

    genome_data.init_data(genome_data_collector=data_collector)

    print("Matrix created.")

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # SAVE THE DATA

    # where to store the genome matrix data
    save_fpath = f"pre_loaded_genome_data/{run_type}_genome_data.zip"
    save_fpath = get_numbered_unique_fpath(save_fpath)

    # set the identifying keys
    identifying_keys = [
        "run_type",
        "use_node_gene_data",
        "use_edge_gene_data",
        "use_edge_weights_data",
        "use_recurrent_edge_gene_data",
        "use_recurrent_edge_weights_data"
    ]

    # make sure this directory exists, so we can output to it
    os.makedirs("pre_loaded_genome_data", exist_ok=True)

    # save the data_storage and reduced data_storage
    genome_data.save_data(
        zip_fpath=save_fpath, identifying_args=get_subset(args, identifying_keys))

if __name__=="__main__":
    main()
