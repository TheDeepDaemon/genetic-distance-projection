"""examples/simple_ga/example.py
This file has example code for visualizing based on a simple genetic algorithm.
"""

"""
 ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
This code block is just here to make is easy to import the GDP module from this particular file.
It is not a part of the example, so you can ignore it.
Note: I know this is a hacky way to do it, but if you are just trying to run an example script,
it requires minimal input on your part.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
"""
 ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
"""
from gdp import GenomeDataCollector, ReducedGenomeData, GenomeVisualizer, reduce_using_pca
from knapsack import Knapsack
import os
import numpy as np


class Genome:

    next_genome_id: int = 0

    def __init__(self, n_genes, mutation_rate, parents=None):
        self.id = Genome.next_genome_id
        Genome.next_genome_id += 1

        self.data = np.zeros(n_genes, dtype=np.bool)
        if parents is not None:

            if len(parents) == 2:
                split_point = np.random.randint(0, n_genes)

                self.data[:split_point] = parents[0].data[:split_point]
                self.data[split_point:] = parents[1].data[split_point:]

            else:
                self.data = parents[0].data

            self.parents = [p.id for p in parents]
        else:
            self.parents = []

        for i in range(n_genes):
            if np.random.random() < mutation_rate:
                self.data[i] = not self.data[i]


def genetic_algorithm_step(
        population: list,
        fitness_function,
        n_genes: int,
        mutation_rate: float,
        n_elites: int):

    new_population = sorted(population, key=lambda x: fitness_function(x), reverse=True)

    elites = new_population[:n_elites]

    offspring_count = len(population) - n_elites
    offspring = []
    for _ in range(offspring_count):
        r1, r2 = np.random.choice(len(new_population), size=2, replace=False)
        parents = [new_population[r1], new_population[r2]]
        offspring.append(Genome(n_genes=n_genes, mutation_rate=mutation_rate, parents=parents))

    return elites + offspring


def run_genetic_algorithm(
        initial_population: list,
        fitness_function,
        num_generations: int,
        n_genes: int,
        mutation_rate: float,
        n_elites: int):
    genome_data_collector = GenomeDataCollector(minimizing_fitness=False)

    population = initial_population

    for i in range(num_generations):
        population = genetic_algorithm_step(
            population=population,
            fitness_function=fitness_function,
            n_genes=n_genes,
            mutation_rate=mutation_rate,
            n_elites=n_elites)

        for member in population:
            info = {
                "fitness": fitness_function(member),
                "id": member.id,
                "data": member.data.copy(),
                "parents": member.parents.copy()}

            genome_data_collector.set_population_member_info(member.id, info)

            for j, gene_value in enumerate(member.data):
                genome_data_collector.set_gene_value(member.id, gene_key=j, gene_value=gene_value)

    return genome_data_collector


def main(knapsack_data):

    knapsack = Knapsack(knapsack_data)

    n_genes = knapsack.num_items
    population_size = 32
    mutation_rate = 0.1
    n_elites = int(population_size // 2)
    n_generations = 100

    initial_population = [
        Genome(n_genes, mutation_rate=mutation_rate)
        for _ in range(population_size)]

    genome_data_collector = run_genetic_algorithm(
        initial_population=initial_population,
        fitness_function=knapsack.evaluate_backpack,
        num_generations=n_generations,
        n_genes=n_genes,
        mutation_rate=mutation_rate,
        n_elites=n_elites)

    reduced_genome_data = ReducedGenomeData.perform_reduction(
        source=genome_data_collector, dim_reduction_function=reduce_using_pca)

    genome_visualizer = GenomeVisualizer(source=reduced_genome_data)

    genome_visualizer.set_colors_by_fitness((0., 0., 1.), (1., 0., 0.))

    source_name = os.path.splitext(os.path.split(knapsack_data)[1])[0]

    genome_visualizer.visualize_genomes2D(
        save_fpath=f"vis-{source_name}", vis_image_type="pdf", node_size=20)


if __name__=="__main__":
    fname = "P01.json"
    main(f"knapsack_data/{fname}")
