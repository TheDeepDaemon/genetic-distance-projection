import numpy as np
import torch


def compare_flattened_distances(gen_distances, pos_distances):
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

    return compare_flattened_distances(genome_distances, position_distances)
