from .genome_data import GenomeData
from typing import Union
import os


class GenomeDataCollector(GenomeData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load(zip_fpath: Union[str, os.PathLike]):
        genome_data_collector: GenomeDataCollector = GenomeDataCollector()
        genome_data_collector._load_from_path(zip_fpath=zip_fpath)
        return genome_data_collector

    def add_population_member(self, genome_id):

        if genome_id not in self._population:
            self._population[genome_id] = dict()

        if genome_id not in self._population_info:
            self._population_info[genome_id] = dict()

    def set_gene_value(self, genome_id: int, gene_key, gene_value):
        self.add_population_member(genome_id=genome_id)
        self._population[genome_id][gene_key] = gene_value

    def add_categorical_gene(self, genome_id: int, gene_variant):
        self.add_population_member(genome_id)
        self._population[genome_id][gene_variant] = True

    def set_population_member_info(self, genome_id, info: dict):
        self.add_population_member(genome_id=genome_id)
        self._population_info[genome_id].update(info)

    def convert_info_to_genes(self, func, key):
        for genome_id, info in self._population_info.items():

            new_genes = func(info[key])

            for gene in new_genes:
                self.add_categorical_gene(genome_id=genome_id, gene_variant=gene)

    def convert_info_to_gene_values(self, func, key):
        for genome_id, info in self._population_info.items():

            new_genes = func(info[key])

            for gene_key, gene_value in new_genes.items():
                self.set_gene_value(genome_id=genome_id, gene_key=gene_key, gene_value=gene_value)
