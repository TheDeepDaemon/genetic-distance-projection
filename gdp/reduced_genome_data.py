from .genome_data import GenomeData
from ._dim_reduction_.nn_reduction import reduce_using_neural_net
import numpy as np
import copy
import json
import os
from typing import Union


# function is implemented here so the genome data collector doesn't need to import numpy
def _convert_genome_data_to_matrix(genome_data: GenomeData):
    """
    Convert the genes in the genome data object to a matrix that you can perform dimensionality reduction on.

    Args:
        genome_data: The genome data to source values from.

    Returns:
        A matrix of the genome data, along with identifiers for the indices.
    """

    # all genome IDs
    genome_ids = genome_data.get_unique_genome_id_list()

    # all unique gene keys
    gene_keys = genome_data.get_unique_gene_key_list()

    # dict for converting gene keys to indices
    gene_index = {gk: j for j, gk in enumerate(gene_keys)}

    # initialize the genes matrix to output to
    genes_matrix = np.zeros((len(genome_ids), len(gene_keys)), dtype=np.float32)

    # iterate through genomes IDs
    for genome_idx, genome_id in enumerate(genome_ids):

        # retrieve the genome data from this genome ID
        genome = genome_data._population[genome_id]

        # iterate through gene keys and their values
        for gene_key, gene_val in genome.items():

            # get the matrix index for this key
            gene_idx = gene_index.get(gene_key)

            # check if that index exists in the dict (if it doesn't, then there isn't a matrix element for this)
            if gene_idx is not None:

                # then set the matrix with this value
                genes_matrix[genome_idx, gene_idx] = gene_val

    return genes_matrix, genome_ids, gene_keys


def _convert_genome_data_to_bag(genome_data: GenomeData):

    # all genome IDs
    genome_ids = genome_data.get_unique_genome_id_list()

    # all unique gene keys
    gene_keys = genome_data.get_unique_gene_key_list()

    # dict for converting gene key to index
    gene_index = {gk: j for j, gk in enumerate(gene_keys)}

    indices = []
    weights = []

    # iterate through genomes IDs
    for genome_id in genome_ids:

        # retrieve the genome data from this genome ID
        genome = genome_data._population[genome_id]

        gene_idx_list = [gene_index[gene_key] for gene_key in genome]
        gene_w_list = [float(gene_weight) for gene_weight in genome.values()]

        indices.append(gene_idx_list)
        weights.append(gene_w_list)

    return genome_ids, indices, weights


class ReducedGenomeData(GenomeData):
    """
    This is a class that includes functionality for encoding genes into a numerical format,
    and for reducing the data down to positions.
    Dimensionality reduction is implemented separately.
    """

    def __init__(
            self,
            source: Union['ReducedGenomeData', 'GenomeData', str, os.PathLike]=None,
            *args,
            **kwargs):
        if isinstance(source, ReducedGenomeData):
            reduced_genome_data: ReducedGenomeData = source
            super().__init__(source=reduced_genome_data, *args, **kwargs)
            self.encoded_genomes = copy.deepcopy(reduced_genome_data.encoded_genomes)
            self.reduced_positions = copy.deepcopy(reduced_genome_data.reduced_positions)
        else:
            self.encoded_genomes = None
            self.reduced_positions = None
            super().__init__(source=source, *args, **kwargs)

    @staticmethod
    def load(zip_fpath: Union[str, os.PathLike]):
        reduced_genome_data: ReducedGenomeData = ReducedGenomeData()
        reduced_genome_data._load_from_path(zip_fpath=zip_fpath)
        return reduced_genome_data

    @staticmethod
    def perform_reduction(source: Union[GenomeData, str, os.PathLike], dim_reduction_function):
        if isinstance(source, GenomeData):
            genome_data = source
        elif isinstance(source, (str, os.PathLike)):
            genome_data = GenomeData.load(zip_fpath=source)
        else:
            raise TypeError("Expected GenomeData instance or path to file.")

        genes_matrix, genome_ids, gene_keys = _convert_genome_data_to_matrix(genome_data=genome_data)

        reduced_genome_data = ReducedGenomeData()
        reduced_genome_data._population = copy.deepcopy(genome_data._population)
        reduced_genome_data._population_info = copy.deepcopy(genome_data._population_info)

        # save the genes
        reduced_genome_data.encoded_genomes = {genome_ids[i]: mat_row for i, mat_row in enumerate(genes_matrix)}

        positions = dim_reduction_function(genes_matrix)

        # save the positions
        reduced_genome_data.reduced_positions = {genome_ids[i]: pos for i, pos in enumerate(positions)}

        return reduced_genome_data

    @staticmethod
    def perform_reduction_nn(source: Union[GenomeData, str, os.PathLike], model_save_fname):
        if isinstance(source, GenomeData):
            genome_data = source
        elif isinstance(source, (str, os.PathLike)):
            genome_data = GenomeData.load(zip_fpath=source)
        else:
            raise TypeError("Expected GenomeData instance or path to file.")

        genome_ids, indices, weights = _convert_genome_data_to_bag(genome_data=genome_data)

        reduced_genome_data = ReducedGenomeData()
        reduced_genome_data._population = copy.deepcopy(genome_data._population)
        reduced_genome_data._population_info = copy.deepcopy(genome_data._population_info)

        gene_keys = genome_data.get_unique_gene_key_list()

        # save the genes
        genes_matrix, _, _ = _convert_genome_data_to_matrix(genome_data=genome_data)
        reduced_genome_data.encoded_genomes = {genome_ids[i]: mat_row for i, mat_row in enumerate(genes_matrix)}

        # perform dimensionality reduction
        positions = reduce_using_neural_net(indices, weights, len(gene_keys), model_save_fname)

        # save the positions
        reduced_genome_data.reduced_positions = {genome_ids[i]: pos for i, pos in enumerate(positions)}

        return reduced_genome_data

    def _save_contents(self, zip_file, **kwargs):
        super()._save_contents(zip_file, **kwargs)

        kwargs.pop("identifying_args")

        kwargs.setdefault('indent', 4)
        encoded_genomes = {k: v.tolist() for k, v in self.encoded_genomes.items()}
        zip_file.writestr("encoded_genomes.json", json.dumps(encoded_genomes, **kwargs))

        reduced_positions = {k: v.tolist() for k, v in self.reduced_positions.items()}
        zip_file.writestr("reduced_positions.json", json.dumps(reduced_positions, **kwargs))

    def _load_contents(self, zip_file):
        super()._load_contents(zip_file)

        with zip_file.open("encoded_genomes.json") as f:
            encoded_genomes = json.loads(f.read().decode('utf-8'))
            encoded_genomes = {int(k): v for k, v in encoded_genomes.items()}
            self.encoded_genomes = {k: np.array(v, dtype=np.float32) for k, v in encoded_genomes.items()}

        with zip_file.open("reduced_positions.json") as f:
            reduced_positions = json.loads(f.read().decode('utf-8'))
            reduced_positions = {int(k): v for k, v in reduced_positions.items()}
            self.reduced_positions = {k: np.array(v, dtype=np.float32) for k, v in reduced_positions.items()}
