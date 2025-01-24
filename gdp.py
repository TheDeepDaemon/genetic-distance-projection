import os
from datetime import datetime
from gene_data_processing import (
    convert_genes_to_numerical, combine_gene_mats, reduce_genome)
from load_data import load_data_path, load_genomes, unpack_data, get_tree_depth
import json
import argparse


def convert_to_numerical(node_genes, edge_genes):
    # take the list of gene IDs and convert to a (float) vector format
    node_genes_matrix, node_genome_id_to_index = convert_genes_to_numerical(node_genes)

    # take the list of gene IDs and convert to a (float) vector format
    edge_genes_matrix, edge_genome_id_to_index = convert_genes_to_numerical(edge_genes)

    # combine the genes matrix, so that both node and edge genes are encoded
    genes_mat, genome_id_to_index = combine_gene_mats(
        genes_mat1=node_genes_matrix,
        genome_id_to_index1=node_genome_id_to_index,
        genes_mat2=edge_genes_matrix,
        genome_id_to_index2=edge_genome_id_to_index)

    return genes_mat, genome_id_to_index


def run_gdp(args):
    # load data
    genome_data_list = load_genomes(data_dir=args.data_path)

    # unpack data
    graph, node_genes, edge_genes, fitnesses = \
        unpack_data(genome_data_list=genome_data_list)

    tree_depth, genome_depths = get_tree_depth(graph)

    genes_mat, genome_id_to_index = convert_to_numerical(
        node_genes=node_genes, edge_genes=edge_genes)

    # find the best fitness, so we can mark the best genome
    best_fitness = float('inf')
    best_genome_id = -1
    for genome_id, fitness in fitnesses.items():
        if fitness < best_fitness:
            best_fitness = fitness
            best_genome_id = genome_id

    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    # set the subdirectory name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cur_run_directory = os.path.join(figures_dir, f"run_results_{current_time}")
    if not os.path.exists(cur_run_directory):
        os.mkdir(cur_run_directory)

    # get the positions by mapping to 2D
    positions = reduce_genome(
        genes_matrix=genes_mat,
        genome_id_to_index=genome_id_to_index,
        reduction_method=args.reduction_method,
        reduced_size=2,
        reduction_type='position',
        best_genome_id=best_genome_id,
        args=args)

    # output to json file
    with open(args.output_fname, 'w', encoding="utf-8") as file:
        json.dump({
            'graph_nodes': list(graph.nodes),
            'graph_edges': list(graph.edges),
            'positions': {k: v.tolist() for k, v in positions.items()},
            'genes_mat': genes_mat.tolist(),
            'genome_id_to_index': genome_id_to_index,
            'genome_depths': genome_depths,
            'tree_depth': tree_depth,
            'fitnesses': fitnesses
        }, file)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run GDP with specified arguments.")

    # Add arguments
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the input data file."
    )
    parser.add_argument(
        "output_fname",
        type=str,
        help="Filename for the output results."
    )
    parser.add_argument(
        "--reduction_method",
        type=str,
        choices=['nn', 'nn1l', 'mds', 'pca', 'svd'],
        default='mds',
        help="Dimensionality reduction method to use."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs."
    )

    # Parse arguments
    args = parser.parse_args()

    # Pass arguments to the `run_gdp` function
    run_gdp(args)
