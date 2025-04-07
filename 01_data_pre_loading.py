from gdp import GenomeMatrix, GenomeDataCollector
from gdp.program_arguments import ProgramArguments
from local_util import get_numbered_unique_fpath
import os


def main():
    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # LOAD PROGRAM ARGUMENTS

    # load the program arguments
    program_args = ProgramArguments()

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # LOAD AND SET GENOME DATA

    data_collector = GenomeDataCollector()
    data_collector.load(f"data/{program_args.run_type}.zip")

    if program_args.use_node_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: ["node_id:{}".format(gn["n"]) for gn in data_entry],
            key="nodes")

    if program_args.use_edge_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: ["edge_id:{}".format(ge["n"]) for ge in data_entry],
            key="edges")

    if program_args.use_edge_weights_data:
        data_collector.convert_info_to_gene_values(
            func=lambda data_entry: {"edge_weight:{}".format(ge["n"]): float(ge["weight"]) for ge in data_entry},
            key="edges")

    if program_args.use_recurrent_edge_gene_data:
        data_collector.convert_info_to_genes(
            func=lambda data_entry: ["recurrent_edge:{}".format(gre["n"]) for gre in data_entry],
            key="recurrent_edges")

    if program_args.use_recurrent_edge_weights_data:
        data_collector.convert_info_to_gene_values(
            func=lambda data_entry: {"recurrent_edge_weight:{}".format(gre["n"]): float(gre["weight"]) for gre in data_entry},
            key="recurrent_edges")

    print("Data encoded.")

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # NUMERICALLY ENCODE INTO A MATRIX

    genome_matrix = GenomeMatrix()

    genome_matrix.init_data(genome_data_collector=data_collector)

    print("Matrix created.")

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # SAVE THE DATA

    # where to store the genome matrix data
    save_fpath = f"pre_loaded_genome_data/{program_args.run_type}_genome_data.zip"
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
    genome_matrix.save_data(
        zip_fpath=save_fpath, identifying_args=program_args.get_subset(identifying_keys))

if __name__=="__main__":
    main()
