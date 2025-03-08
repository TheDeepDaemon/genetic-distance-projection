import numpy as np
import json
import os
import networkx as nx
from collections.abc import Sequence
from typing import Tuple, Optional


def load_genomes(data_dir: str):
    """
    Load genomes from json files in a folder.

    :param data_dir: The directory to retrieve the json files from.
    :return: The data_storage from the run.
    """
    json_files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if (os.path.isfile(os.path.join(data_dir, fname)) and
         (os.path.splitext(fname)[-1] == ".json"))]

    genomes = []

    for fpath in json_files:
        with open(fpath, 'r', encoding="utf-8") as file:
            genomes.append(json.load(file))

    return genomes


def unpack_data(genome_data_list) -> Optional[
    Tuple[
        nx.DiGraph,
        dict[int, list[int]],
        dict[int, list[int]],
        dict[int, float],
        list[int]]]:
    """
    Unpack the data_storage that has been loaded.

    :param genome_data_list:
    :return:
    """

    # create a directed graph
    graph = nx.DiGraph()

    # all genome IDs
    genome_id_list = []

    # create the dicts
    node_genes = dict()
    edge_genes = dict()
    fitnesses = dict()

    for genome_data in genome_data_list:

        nodes = genome_data['nodes']
        edges = genome_data['edges']
        genome_id = genome_data['generation_number']
        parents = genome_data['parents']
        fitness = genome_data['fitness']

        genome_id_list.append(genome_id)

        # set the attributes for this node id
        node_genes[genome_id] = nodes
        edge_genes[genome_id] = edges
        fitnesses[genome_id] = fitness

        # add the node to make sure it is in the graph
        if not graph.has_node(genome_id):
            graph.add_node(genome_id)

        # check parents
        assert isinstance(parents, Sequence)

        # iterate through all parents
        for p_id in parents:

            # don't store self-connections
            if genome_id != p_id:

                # add the edge now
                graph.add_edge(p_id, genome_id)

    nodes_to_remove = set(graph.nodes) - set(genome_id_list)

    graph.remove_nodes_from(nodes_to_remove)

    # track global best
    global_best_fitness = float('inf')
    line_of_succession = []

    for genome_id in sorted(genome_id_list):
        fitness = fitnesses[genome_id]
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            line_of_succession.append(genome_id)

    return graph, node_genes, edge_genes, fitnesses, line_of_succession


def _gene_list_to_numerical(gene_ids, gene_index_dict):
    """
    Takes a gene list and outputs a one-hot vector.
    Args:
        gene_ids (list[[node_inon_t, edge_inon_t]]): A list of node or edge IDs.
        gene_index_dict (dict[[node_inon_t, edge_inon_t], int]): A dict that maps node or edge id to an index.

    Returns:
        np.ndarray: A one-hot encoded vector.
    """
    arr = np.zeros(len(gene_index_dict), dtype=float)
    for gene_id in gene_ids:
        arr[gene_index_dict[gene_id]] = 1

    return arr


def convert_genes_to_numerical(genome_genes: dict):
    """
    Convert all genome node or edge lists into a numerical format for further processing.

    Args:
        genome_genes (dict[int, list[[node_inon_t, edge_inon_t]]]):
            A dict that maps a genome generation number to the list of node or edge genes.

    Returns:
        dict: A dict of all node generation numbers to numerical vectors.
    """

    # we create a set of all node IDs used here
    all_unique_ids = set()

    for array in genome_genes.values():
        all_unique_ids.update(array)

    # we get a mapping of gene to index
    gene_index_dict = {gene_id: i for i, gene_id in enumerate(all_unique_ids)}

    # convert all lists of node genes into a numerical format
    genome_genes_numerical = {
        genome_id: _gene_list_to_numerical(genes, gene_index_dict)
        for genome_id, genes in genome_genes.items()}

    # matrix where the rows are genomes, and the columns are genes
    genes_matrix = np.zeros((len(genome_genes_numerical), len(all_unique_ids)), dtype=float)
    genome_id_to_index = dict()

    for i, (genome_id, genes) in enumerate(genome_genes_numerical.items()):
        genes_matrix[i] = genes
        genome_id_to_index[genome_id] = i

    return genes_matrix, genome_id_to_index