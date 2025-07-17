import numpy as np


class Genome:
    """
    The Genome class, which contains gene values as a vector, and some other things, like ID.
    """

    next_genome_id: int = 0

    def __init__(self, fitness_function, n_genes, mutation_rate, parents=None):
        # each ID is just incremented from the last
        self.id = Genome.next_genome_id
        Genome.next_genome_id += 1

        # data is zero by default
        self.data = np.zeros(n_genes, dtype=np.bool)

        # if there are parents source the gene values from them
        if parents is not None:

            # if there are two parents, do crossover
            if len(parents) == 2:
                split_point = np.random.randint(0, n_genes)

                self.data[:split_point] = parents[0].data[:split_point]
                self.data[split_point:] = parents[1].data[split_point:]

            else:
                # we are expecting only one parent here
                self.data = parents[0].data

            # track the parents
            self.parents = [p.id for p in parents]
        else:
            self.parents = []

        # mutate the genes
        for i in range(n_genes):
            if np.random.random() < mutation_rate:
                self.data[i] = not self.data[i]

        # set the fitness once, on creation
        self.fitness = fitness_function(self.data)


def genetic_algorithm_step(
        population: list,
        fitness_function,
        n_genes: int,
        mutation_rate: float,
        n_elites: int):
    """
    One step of a genetic algorithm. Call this repeatedly to run the GA.

    Args:
        population: The input population.
        fitness_function: The fitness function for evaluation.
        n_genes: The number of genes.
        mutation_rate: The mutation rate.
        n_elites: The number of elites.

    Returns:
        The updated population.
    """

    # sort based on fitness
    new_population = sorted(population, key=lambda x: x.fitness, reverse=True)

    # get the elites from the best of the population
    elites = new_population[:n_elites]

    # set the number of offspring
    offspring_count = len(population) - n_elites

    # list for collecting offspring
    offspring = []
    for _ in range(offspring_count):
        # choose two parents randomly
        r1, r2 = np.random.choice(len(elites), size=2, replace=False)
        parents = [elites[r1], elites[r2]]

        # create a new genome with them
        new_genome = Genome(
            fitness_function=fitness_function,
            n_genes=n_genes,
            mutation_rate=mutation_rate,
            parents=parents)

        # add the genome to the population
        offspring.append(new_genome)

    return elites + offspring
