import os.path

import networkx as nx
import matplotlib.pyplot as plt


def visualize_genomes2D(
        save_fpath: str,
        graph: nx.DiGraph,
        positions: dict,
        genome_colors,
        legend_handles,
        **kwargs):
    """
    Perform 2D visualizations, save to an image file.

    Args:
        save_fpath: The filepath to save the data to.
        graph: The graph to show.
        positions: The positions of the nodes.
        genome_colors: The node colors for the genomes.
        legend_handles: The legend handles to use for the plot.
    """

    genome_ids = set(positions.keys())
    _pos_keys = set([int(k) for k in positions.keys()])

    assert (set(genome_ids) == _pos_keys)
    assert (set(genome_ids) == set(genome_colors.keys()))

    kwargs.setdefault("title", "Population Graph")

    node_size = kwargs["node_size"]
    arrow_size = kwargs["arrow_size"]
    line_width = kwargs["line_width"]
    title = kwargs["title"]

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
    plt.title(title)

    img_type = kwargs.get("vis_image_type")
    if img_type:
        root, ext = os.path.splitext(save_fpath)
        if ext.lower() != f".{img_type.lower()}":
            save_fpath = f"{save_fpath}.{img_type}"

    plt.savefig(save_fpath, format=img_type if img_type else None)
    print(f"Saved figure to {save_fpath}")

    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    return x_limits, y_limits
