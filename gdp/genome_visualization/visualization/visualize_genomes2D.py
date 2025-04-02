import networkx as nx
import matplotlib.pyplot as plt
from ...genome_data import GenomeDataCollector


def visualize_genomes2D(
        save_fpath: str, genome_data_collector: GenomeDataCollector, genome_colors, args, legend_handles):
    """
    Perform 2D visualizations, save to an image file.

    Args:
        save_fpath: The filepath to save the data to.
        genome_data_collector: The genome data to use.
        genome_colors: The node colors for the genomes.
        args: Passed arguments and their keywords.
        legend_handles: The legend handles to use for the plot.
    """

    genome_ids = genome_data_collector.get_unique_genome_id_list()
    positions = genome_data_collector.get_genome_attribute_by_key("reduced_position")
    _pos_keys = set([int(k) for k in positions.keys()])

    assert (set(genome_ids) == _pos_keys)
    assert (set(genome_ids) == set(genome_colors.keys()))

    node_size = args["node_size"]
    arrow_size = args["arrow_size"]
    line_width = args["line_width"]

    graph = genome_data_collector.make_graph()

    colors = [genome_colors[int(node)] for node in graph.nodes]

    pos = nx.spring_layout(graph, pos=positions, fixed=genome_ids)

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

    if legend_handles is not None:
        plt.legend(handles=legend_handles, title="Node Colors")

    plt.savefig(save_fpath)
    print(f"Saved figure to {save_fpath}")

    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    return x_limits, y_limits
