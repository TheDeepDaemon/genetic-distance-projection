import networkx as nx

def make_relations_graph(genome_data):
    """
    Build the graph that represents all relations between genomes.

    Args:
        genome_data: The dictionary of all genome data.

    Returns:
        The relations graph.
    """
    graph = nx.DiGraph()

    for gid in genome_data:
        graph.add_node(gid)

    for gid, data in genome_data.items():
        for parent in data['parents']:
            if parent in genome_data:
                graph.add_edge(int(parent), gid)

    return graph
