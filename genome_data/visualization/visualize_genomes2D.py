import networkx as nx
import matplotlib.pyplot as plt


def visualize_genomes2D(args, seed=42):
    """
    Perform 2D visualizations, save to an image file.

    Args:
        args: Passed arguments and their keywords.
        seed: The random seed.
    """

    genome_ids_set = set(args["genome_ids"])
    genome_colors = args["genome_colors"]
    positions = args["positions"]

    assert (genome_ids_set == set(positions.keys()))
    assert (genome_ids_set == set(genome_colors.keys()))

    node_size = args["node_size"]
    arrow_size = args["arrow_size"]
    line_width = args["line_width"]

    graph = args["graph"]

    colors = [genome_colors[int(node)] for node in graph.nodes]

    pos = nx.spring_layout(graph, pos=positions, fixed=genome_ids_set, seed=seed)

    plt.figure(figsize=(7, 7))

    # draw nodes with fill color on top
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_color=colors,
        node_size=node_size,
        arrows=True,
        arrowsize=arrow_size,
        width=line_width)

    legend_handles = args["legend_handles"]

    if legend_handles is not None:
        plt.legend(handles=legend_handles, title="Node Colors")

    save_fpath = args["save_fpath"]
    plt.savefig(save_fpath)
    print(f"Saved figure to {save_fpath}")

    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    return x_limits, y_limits
