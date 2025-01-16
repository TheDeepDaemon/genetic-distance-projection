import json
import os
import networkx as nx
from collections.abc import Sequence
from typing import List, Tuple, Optional


def load_genomes(data_dir: str):
    json_files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if (os.path.isfile(os.path.join(data_dir, fname)) and
         (os.path.splitext(fname)[-1] == ".json"))]

    genomes = []

    for fpath in json_files:
        with open(fpath, 'r') as file:
            genomes.append(json.load(file))

    return genomes


def unpack_data(genome_data_list) -> Optional[
    Tuple[
        nx.DiGraph,
        dict[int, list[int]],
        dict[int, list[int]],
        dict[int, float]]]:

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

        assert isinstance(parents, Sequence)

        # iterate through all parents
        for p_id in parents:

            # don't store self-connections
            if genome_id != p_id:
                # add the edge now
                graph.add_edge(p_id, genome_id)

    nodes_to_remove = set(graph.nodes) - set(genome_id_list)

    graph.remove_nodes_from(nodes_to_remove)

    return graph, node_genes, edge_genes, fitnesses


def get_tree_depth(graph):

    tree_depth = 0

    ancestors = []
    cur_gen = [1]

    genome_depths = dict()

    while True:
        next_gen = []
        for member in cur_gen:
            genome_depths[member] = tree_depth
            neighbors = [nb for nb in graph.neighbors(member) if nb not in ancestors]
            next_gen += neighbors
        ancestors = ancestors + cur_gen
        tree_depth += 1

        if len(next_gen) == 0:
            break
        else:
            cur_gen = next_gen

    return tree_depth, genome_depths


def load_data_path():
    with open("data-dir-path", 'r') as file:
        return file.read()


if __name__=="__main__":
    data_dir = load_data_path()
    genomes = load_genomes(data_dir=data_dir)
    graph, node_genes, edge_genes, fitnesses = unpack_data(genomes)
    tree_depth = get_tree_depth(graph)
    print(f"tree depth: {tree_depth}")
