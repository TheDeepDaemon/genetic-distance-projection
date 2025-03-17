import numpy as np
import networkx as nx


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
