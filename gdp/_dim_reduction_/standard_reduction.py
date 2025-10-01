import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import warnings


def reduce_using_pca(genes_matrix: np.ndarray, reduced_size: int=2):
    """
    Reduce the genes matrix using PCA.

    Args:
        genes_matrix (np.ndarray): A 2D matrix of genes.
        reduced_size (int): The reduced size, usually 2 or 3.

    Returns:
        np.ndarray: The reduced matrix.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pca = PCA(n_components=reduced_size)
        return pca.fit_transform(genes_matrix)


def reduce_using_svd(genes_matrix: np.ndarray, reduced_size: int=2):
    """
    Reduce the genes matrix using SVD.

    Args:
        genes_matrix (np.ndarray): A 2D matrix of genes.
        reduced_size (int): The reduced size, usually 2 or 3.

    Returns:
        np.ndarray: The reduced matrix.
    """
    U, S, VT = np.linalg.svd(genes_matrix, full_matrices=False)
    return genes_matrix @ VT[:reduced_size, :].T


def reduce_using_mds(genes_matrix: np.ndarray, reduced_size: int=2):
    """
    Reduce the genes matrix using MDS.

    Args:
        genes_matrix (np.ndarray): A 2D matrix of genes.
        reduced_size (int): The reduced size, usually 2 or 3.

    Returns:
        np.ndarray: The reduced matrix.
    """
    mds = MDS(n_components=reduced_size, dissimilarity="precomputed")

    distance_mat = pairwise_distances(genes_matrix, metric='euclidean')

    return mds.fit_transform(distance_mat)


def reduce_using_t_sne(genes_matrix: np.ndarray, reduced_size: int=2, **kwargs):
    tsne = TSNE(n_components=reduced_size)
    return tsne.fit_transform(genes_matrix)
