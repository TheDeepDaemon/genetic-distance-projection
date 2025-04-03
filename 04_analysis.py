import numpy as np
import torch
from gdp import GenomeMatrix
import os
from program_arguments import ProgramArguments


def compare_distances(gen_distances, pos_distances):
    projection = (np.dot(gen_distances, pos_distances) / np.dot(pos_distances, pos_distances)) * pos_distances
    return np.linalg.norm(gen_distances - projection)


def get_pwise_distances(mat: np.ndarray):

    # get torch tensor
    tens = torch.from_numpy(mat)

    # get distance matrix using pytorch (it's faster)
    dist_tens = torch.cdist(tens, tens, p=2)

    # get the lower diagonal
    dist_mat = torch.tril(dist_tens, diagonal=-1).numpy()

    # convert to a 1D array and return
    rows, cols = np.tril_indices_from(dist_mat, k=-1)
    return dist_mat[rows, cols]


def get_all_distances(
        genome_matrix: np.ndarray,
        positions_matrix: np.ndarray):

    genome_distances = get_pwise_distances(genome_matrix)

    position_distances = get_pwise_distances(positions_matrix)

    return compare_distances(genome_distances, position_distances)


if __name__=="__main__":

    # load the program arguments
    program_args = ProgramArguments()

    identifying_keys = [
        "reduction_type",
        "run_type",
        "use_node_gene_data",
        "use_edge_gene_data",
        "use_edge_weights_data",
        "use_recurrent_edge_gene_data",
        "use_recurrent_edge_weights_data"
    ]

    # find the genome data to use
    load_fname = GenomeMatrix.find_latest_genome_data(
        data_dir="reduced_genome_data",
        identifying_args=program_args.get_subset(identifying_keys))

    # create the genome data storage class to be used for visuals
    genome_data = GenomeMatrix()

    # load the processed data storage from the directory
    genome_data.load_data(
        zip_fpath=os.path.join("reduced_genome_data", load_fname),
        identifying_args=program_args.get_subset(identifying_keys))

    dist = get_all_distances(
        genome_matrix=genome_data.genome_data_mat,
        positions_matrix=genome_data.position_data)

    for k, v in program_args.args.items():
        print(f"{k}: {v}")

    print()
    print("dist: ", dist)

