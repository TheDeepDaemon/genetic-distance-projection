import json
import numpy as np
import zipfile


class GenomeDataCollector:

    def __init__(self):
        self._population = dict()
        self._population_info = dict()

    def add_population_member(self, genome_id):
        """
        Add a member to the population.

        Args:
            genome_id: The ID of the member to add.
        """
        if genome_id not in self._population:
            self._population[genome_id] = dict()

        if genome_id not in self._population_info:
            self._population_info[genome_id] = dict()

    @staticmethod
    def _check_if_gene_value_valid(gene_value):
        """
        Check to see if the value of a gene being set can be used.

        Args:
            gene_value: The value of the gene we are trying to set.

        Returns:
            Whether the gene value is valid.
        """
        return (
                (
                        isinstance(gene_value, (bool, np.bool_)) or
                        isinstance(gene_value, (float, np.floating))
                ) or
                (
                        isinstance(gene_value, (int, np.integer)) and
                        ((gene_value == 1) or (gene_value == 0))
                ))

    def set_gene_value(self, genome_id: int, gene_key, gene_value):
        """
        Set the value of a gene.

        Args:
            genome_id: The genome the gene should belong to.
            gene_key: What gene to set.
            gene_value: The value of the gene.
        """

        if GenomeDataCollector._check_if_gene_value_valid(gene_value):

            if genome_id not in self._population:
                self.add_population_member(genome_id)

            if isinstance(gene_value, (bool, np.bool_)):
                gene_value = int(gene_value)

            self._population[genome_id][gene_key] = gene_value
        else:
            raise ValueError(
                f"Gene value: {gene_value} is not of a valid type. "
                f"Maybe try using the function \'add_categorical_gene\' instead.")

    def add_categorical_gene(self, genome_id: int, gene_variant):
        """
        Add a gene that is either there or it isn't.

        Args:
            genome_id: The genome the gene should belong to.
            gene_variant: The key or identity of the gene.
        """

        if genome_id not in self._population:
            self.add_population_member(genome_id)

        self._population[genome_id][gene_variant] = True

    def save(self, zip_fpath):
        """
        Save the population to a json file.

        Args:
            zip_fpath: The filepath to save it to.
        """
        with zipfile.ZipFile(zip_fpath, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("population.json", json.dumps(self._population, indent=4))
            zipf.writestr("population_info.json", json.dumps(self._population_info, indent=4))

    def load(self, zip_fpath):
        """
        Load the population from a json file.

        Args:
            zip_fpath: The filepath to load from.
        """
        with zipfile.ZipFile(zip_fpath, "r") as zipf:
            with zipf.open("population.json") as f:
                population = json.loads(f.read().decode('utf-8'))
                self._population = {int(k): v for k, v in population.items()}

            with zipf.open("population_info.json") as f:
                population_info = json.loads(f.read().decode('utf-8'))
                self._population_info = {int(k): v for k, v in population_info.items()}

    def get_unique_genome_id_list(self):
        pop_keys = self._population.keys()
        assert (pop_keys == self._population_info.keys())
        return list(pop_keys)

    def get_unique_gene_keys(self):
        """
        Get a list of all gene keys in the population.

        Returns:
            All unique gene keys.
        """
        unique_gene_keys = set()
        for genome in self._population.values():
            assert(isinstance(genome, dict))
            unique_gene_keys.update(set(genome.keys()))
        return unique_gene_keys

    def get_dtype(self):
        """
        Get the dtype to use when constructing the matrix.
        If there are any floating point values, the whole matrix is floating point.
        Otherwise, boolean is used.

        Returns:
            Either boolean or floating point.
        """
        for genome in self._population.values():
            for _, val in genome.items():
                if isinstance(val, (float, np.floating)):
                    return np.float32
        return np.bool_

    def convert_to_matrix(self):
        """
        Convert the contents of the populations to a matrix for further processing.

        Returns:
            The matrix as a numpy array.
        """

        dtype = self.get_dtype()

        genome_ids = list(self._population.keys())
        gene_keys = list(self.get_unique_gene_keys())

        genes_matrix = np.zeros((len(genome_ids), len(gene_keys)), dtype=dtype)

        for i, gid in enumerate(genome_ids):
            for j, gk in enumerate(gene_keys):
                if gk in self._population[gid]:
                    genes_matrix[i, j] = self._population[gid][gk]
                else:
                    genes_matrix[i, j] = dtype(0)

        return genes_matrix, genome_ids, gene_keys

    def show_abbreviated(self):
        """
        Show the population as a json. Only genes with nonzero values need to be included.
        """
        pop_json = json.dumps(self._population, indent=4)
        gene_keys = list(self.get_unique_gene_keys())
        print("population:")
        print(pop_json)
        print(f"\nunique gene keys: {gene_keys}")

    def show_contents(self, indent: int=4):
        """
        Show the population with all values shown for all genomes.

        Args:
            indent: Indent while showing gene values.
        """
        genes_matrix, genome_ids, gene_keys = self.convert_to_matrix()
        indent_str = ' ' * indent

        for i, gid in enumerate(genome_ids):
            print(f"Genome {gid}:")
            for j, gk in enumerate(gene_keys):
                print(f"{indent_str}{gk}: {genes_matrix[i, j]}")

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

    def get_genome_attribute_by_key(self, *args):
        if len(args) == 1:
            key = args[0]
            return {genome_id: info[key] for genome_id, info in self._population_info.items()}
        elif len(args) == 2:
            outer_key, inner_key = args
            result = dict()
            for genome_id, info in self._population_info.items():
                inner_list = []
                for item in info[outer_key]:
                    inner_list.append(item[inner_key])

                result[genome_id] = inner_list
            return result
        else:
            raise TypeError(f"get_genome_attribute_by_key expected 1 or 2 arguments, got {len(args)}")

    def get_global_best(self, fitness_key="fitness"):

        genome_id_list = self.get_unique_genome_id_list()
        genome_id_list.sort()

        fitnesses = self.get_genome_attribute_by_key(fitness_key)

        current_best_genome = None
        current_best_fitness = float('inf')

        for genome_id in genome_id_list:
            gf = fitnesses[genome_id]
            if gf < current_best_fitness:
                current_best_fitness = gf
                current_best_genome = genome_id

        return current_best_genome

    def get_line_of_succession(self, fitness_key="fitness"):
        line_of_succession = []

        genome_id_list = self.get_unique_genome_id_list()
        genome_id_list.sort()

        fitnesses = self.get_genome_attribute_by_key(fitness_key)

        current_best_fitness = float('inf')

        for genome_id in genome_id_list:
            gf = fitnesses[genome_id]
            if gf < current_best_fitness:
                current_best_fitness = gf
                line_of_succession.append(genome_id)

        return line_of_succession
