import networkx as nx

def get_input_output_nodes(graph: nx.DiGraph):
    """
    Get the input nodes and the output nodes of this neural network graph.

    Returns:
        A tuple containing the input nodes and output nodes.
    """
    input_nodes = list(graph.nodes)

    for node in graph.nodes:
        for edge in graph.edges:
            _, n_out = edge
            if node == n_out:
                input_nodes.remove(node)
                break

    output_nodes = list(graph.nodes)

    for node in graph.nodes:
        for edge in graph.edges:
            n_in, n_out = edge
            if node == n_in:
                output_nodes.remove(node)
                break

    return input_nodes, output_nodes
