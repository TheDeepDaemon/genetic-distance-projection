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
from genome import Genome, genetic_algorithm_step
import os


def run_genetic_algorithm(
        initial_population: list,
        fitness_function,
        num_generations: int,
        n_genes: int,
        mutation_rate: float,
        n_elites: int):
    """
    Run the genetic algorithm, and return the genome data that has been collected.

    Args:
        initial_population: The initial population to evolve.
        fitness_function: The fitness function to evaluate genomes with.
        num_generations: The number of generations to run it for.
        n_genes: The number of genes in each genome.
        mutation_rate: How frequently mutations should happen.
        n_elites: The number of elites.

    Returns:
        The genome data object.
    """

    # collector for data
    genome_data_collector = GenomeDataCollector(minimizing_fitness=False)

    # initialize the population
    population = initial_population

    # run the GA for num_generations
    for i in range(num_generations):

        # update the population
        population = genetic_algorithm_step(
            population=population,
            fitness_function=fitness_function,
            n_genes=n_genes,
            mutation_rate=mutation_rate,
            n_elites=n_elites)

        # track the genome data
        track_genomes(population, genome_data_collector, i)

    return population, genome_data_collector


def track_genomes(population, genome_data_collector, generation):
    """
    Track the genome data for the members of the population.

    Args:
        population: The list of genomes.
        genome_data_collector: The genome data collector to use.
        generation: The current generation this is called from.
    """
    # iterate over the population members
    for member in population:

        # has this population member already been collected?
        collected = genome_data_collector.get_if_collected(member.id)

        # if it hasn't been collected...
        if not collected:

            # get the genome info
            info = {
                "id": member.id,
                "fitness": member.fitness,
                "data": member.data.copy(),
                "parents": member.parents.copy(),
                "generation_created": generation
            }

            # set the info for this member
            genome_data_collector.set_population_member_info(member.id, info)

            # collect the genes too
            collect_genes_example1(member, genome_data_collector)


def collect_genes_example1(genome, genome_data_collector):
    """
    The first example of how you can track the gene values.
    """
    for j, gene_value in enumerate(genome.data):
        if gene_value:
            genome_data_collector.add_categorical_gene(genome.id, gene_variant=j)


def collect_genes_example2(genome, genome_data_collector):
    """
    The second example of how you can track the gene values.
    """
    for j, gene_value in enumerate(genome.data):
        genome_data_collector.set_gene_value(genome.id, gene_key=j, gene_value=gene_value)


def main(knapsack_data):

    # initialize the knapsack we are solving
    knapsack = Knapsack(knapsack_data)

    # set the fitness function
    fitness_function = knapsack.evaluate_backpack

    # initialize parameters of this run
    n_genes = knapsack.num_items
    population_size = 32
    mutation_rate = 0.1
    n_elites = int(population_size // 2)
    n_generations = 100

    # generate the initial population
    initial_population = [
        Genome(
            fitness_function=fitness_function,
            n_genes=n_genes,
            mutation_rate=mutation_rate)
        for _ in range(population_size)]

    # run the GA
    _, genome_data_collector = run_genetic_algorithm(
        initial_population=initial_population,
        fitness_function=fitness_function,
        num_generations=n_generations,
        n_genes=n_genes,
        mutation_rate=mutation_rate,
        n_elites=n_elites)

    # perform dimensionality reduction on the data collected
    reduced_genome_data = ReducedGenomeData.perform_reduction(
        source=genome_data_collector, dim_reduction_function=reduce_using_pca)

    # create the visualizer with this data
    genome_visualizer = GenomeVisualizer(source=reduced_genome_data)

    # color the nodes of the graph
    genome_visualizer.set_colors_by_fitness((0., 0., 1.), (1., 0., 0.))

    # set the name of the output
    source_name = os.path.splitext(os.path.split(knapsack_data)[1])[0]

    # verify that the output directory exists
    os.makedirs("output", exist_ok=True)

    # visualize the genomes and output data
    genome_visualizer.visualize_genomes2D(
        save_fpath=f"output/vis-{source_name}", vis_image_type="pdf", node_size=20)


if __name__=="__main__":
    fname = "P01.json"
    main(f"knapsack_data/{fname}")
