import json
import numpy as np
from gene_data_processing import get_all_distances, compare_distances


def decode_dict(d):
    return {int(k): v for k, v in d.items()}


def analyze_distances(input_fname):

    # output to json file
    with open(input_fname, 'r') as file:
        contents = json.load(file)
        graph_nodes = contents['graph_nodes']
        graph_edges = contents['graph_edges']
        positions = decode_dict(contents['positions'])
        positions = {k: np.array(v) for k, v in positions.items()}
        genes_mat = np.array(contents['genes_mat'])
        genome_id_to_index = decode_dict(contents['genome_id_to_index'])
        genome_depths = decode_dict(contents['genome_depths'])
        tree_depth = contents['tree_depth']
        fitnesses = decode_dict(contents['fitnesses'])

    gen_distances, nn_distances = get_all_distances(
        genes_mat=genes_mat, positions=positions, genome_id_to_index=genome_id_to_index)

    nn_comp_dist = compare_distances(gen_distances, nn_distances)
    print(f"scaled distances: {nn_comp_dist}")
    print()


if __name__=="__main__":
    analyze_distances("output.json")
