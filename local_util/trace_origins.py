
def trace_gene_origin(genes_dict, parents_dict, genome_id, gene):
    """
    Finds the path to the genome this gene originates from.

    Args:
        genes_dict: The dictionary of genome ID to genes list.
        parents_dict: The dictionary of genome ID to parents list.
        genome_id: The genome ID to start the search from.
        gene: The gene to trace the path of.

    Returns:
        The path from genome to the originator of a particular gene.
    """

    # check to see if this gene is in the genome we are starting from
    assert (gene in genes_dict[genome_id])

    trace = []

    next_genomes = [genome_id]

    while len(next_genomes) > 0:
        cur_genome_id = next_genomes[0]
        trace.append(cur_genome_id)
        parents = parents_dict[cur_genome_id]
        next_genomes = [p for p in parents if (p in genes_dict) and (gene in genes_dict[p])]

    return trace
