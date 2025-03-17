import networkx as nx


def get_genome_graph(genome_data):
    """
    Take a dict of genome data and convert it into a graph.

    Args:
        genome_data: The dictionary of genome data.

    Returns:
        Graph with nodes and edges included.
    """
    graph = nx.DiGraph()

    for node_data in genome_data["nodes"]:
        graph.add_node(node_data['n'])

    for edge_data in genome_data["edges"]:
        graph.add_edge(edge_data['in'], edge_data['on'])

    return graph
