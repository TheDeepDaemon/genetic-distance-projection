
def get_global_best(fitnesses: dict):

    genome_id_list = list(fitnesses.keys())
    genome_id_list.sort()

    current_best_genome = None
    current_best_fitness = float('inf')

    for genome_id in genome_id_list:
        gf = fitnesses[genome_id]
        if gf < current_best_fitness:
            current_best_fitness = gf
            current_best_genome = genome_id

    return current_best_genome


def get_line_of_succession(fitnesses: dict):
    line_of_succession = []

    genome_id_list = list(fitnesses.keys())
    genome_id_list.sort()

    current_best_fitness = float('inf')

    for genome_id in genome_id_list:
        gf = fitnesses[genome_id]
        if gf < current_best_fitness:
            current_best_fitness = gf
            line_of_succession.append(genome_id)

    return line_of_succession
