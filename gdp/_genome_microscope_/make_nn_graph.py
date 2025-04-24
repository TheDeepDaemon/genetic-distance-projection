import networkx as nx
import numpy as np


def get_genome_graph(nodes, edges):
    """
    Take a dict of genome data and convert it into a graph.

    Args:
        nodes: The nodes of the neural network.
        edges: The edges of the neural network.

    Returns:
        Graph with nodes and edges included.
    """
    graph = nx.DiGraph()

    for n in nodes:
        graph.add_node(n)

    for e in edges:
        graph.add_edge(*e)

    return graph


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


def find_node_depths(graph: nx.DiGraph, input_nodes: list[int]):
    """
    Find the maximum depths of each of the nodes.

    Args:
        graph: The neural network directed graph.
        input_nodes: The nodes that serve as the inputs.

    Returns:
        A dictionary containing the depth of each node.
    """

    # convert edges to a matrix
    adj_mat = nx.to_numpy_array(graph).T.astype(dtype=np.int64)
    node_order = list(graph.nodes())

    # initialize a signal to start passing
    signal = [(1 if n_id in input_nodes else 0) for n_id in node_order]
    signal = np.array(signal, dtype=np.int64)

    # dictionary to keep track of node depths
    node_depths = {node: None for node in graph.nodes}

    max_depth = len(node_order)

    for cur_depth in range(max_depth):

        # iterate over all nodes and update depths
        for idx, val in enumerate(signal):

            # check if the signal is passing through this node
            if val != 0:

                # get the node ID
                n_id = node_order[idx]

                # set the depth of the node to the latest signal passing
                node_depths[n_id] = cur_depth

        # pass signal to next nodes
        signal = np.dot(adj_mat, signal)

    return node_depths


def make_nn_graph(nodes, edges, recurrent_edges, x_spacing, y_spacing):

    graph = get_genome_graph(nodes, edges)

    input_nodes, output_nodes = get_input_output_nodes(graph)

    node_depths = find_node_depths(graph, input_nodes=input_nodes)

    max_depth = max(node_depths.values())

    y_positions = dict()

    for i in range(max_depth + 1):
        this_layer = [node for node in graph.nodes if (node_depths[node] == i)]

        y_range = len(this_layer) * y_spacing

        cur_y = -y_range / 2
        for node in this_layer:
            y_positions[node] = cur_y
            cur_y += y_spacing

    x_positions = dict()

    for node, depth in node_depths.items():
        x_positions[node] = depth * x_spacing

    positions = {node: (x_positions[node], y_positions[node]) for node in graph.nodes}

    # add in the recurrent edges afterwards
    for re in recurrent_edges:
        graph.add_edge(*re)

    return graph, positions
