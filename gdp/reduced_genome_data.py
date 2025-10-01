from .genome_data import GenomeData
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
    genome_ids = genome_data.get_unique_genome_id_list()
    gene_keys = genome_data.get_unique_gene_key_list()

    gene_index = {gk: j for j, gk in enumerate(gene_keys)}
    genes_matrix = np.zeros((len(genome_ids), len(gene_keys)), dtype=np.float32)

    for i, gid in enumerate(genome_ids):
        genome = genome_data._population[gid]
        for k, val in genome.items():
            j = gene_index.get(k)
            if j is not None:
                genes_matrix[i, j] = val

    return genes_matrix, genome_ids, gene_keys


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
