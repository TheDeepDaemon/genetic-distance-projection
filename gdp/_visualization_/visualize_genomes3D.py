import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .arrow3d import Arrow3D
import io
import os
import imageio.v2 as imageio


def _visualize_genomes3D(
        graph: nx.DiGraph,
        positions: dict,
        genome_colors: dict,
        traces: list=None,
        dimmer_list: list=None,
        **kwargs):
    """
    Perform the 3D _visualization_, save to a GIF.

    Args:
        genome_data_collector: The genome data.
        genome_colors: The node colors for the genomes.
        args: Passed arguments and their keywords.
        traces: The list of paths to trace, and their color maps.
        dimmer_list: A list of nodes to dim.
        title: The title to show in the 3D graph plot.
    """

    # set defaults
    kwargs.setdefault("title", "3D Population Graph")
    kwargs.setdefault("t_positions", None)

    # program arguments
    node_size = kwargs["node_size"]
    arrow_size = kwargs["arrow_size"]
    line_width = kwargs["line_width"]
    line_alpha = kwargs["line_alpha_3D"]
    title = kwargs["title"]
    t_positions = kwargs["t_positions"]

    # create the figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    for node_gid in graph.nodes:
        x, y = positions[node_gid]

        alpha = kwargs["node_alpha_on_3D"]

        if node_gid in dimmer_list:
            alpha = kwargs["node_alpha_off_3D"]

        ax.plot(
            node_gid, x, y,
            marker='o',
            markersize=node_size,
            color=genome_colors[node_gid],
            alpha=alpha)

    for edge in graph.edges:
        from_node_gid, to_node_gid = edge

        start_pos2d = positions[from_node_gid]

        end_pos2d = positions[to_node_gid]

        if t_positions is None:
            from_tpos = from_node_gid
            to_tpos = to_node_gid
        else:
            from_tpos = t_positions[from_node_gid]
            to_tpos = to_node_gid[to_node_gid]

        a = Arrow3D(
            [from_tpos, to_tpos],
            [start_pos2d[0], end_pos2d[0]],
            [start_pos2d[1], end_pos2d[1]],
            mutation_scale=arrow_size,
            lw=line_width,
            arrowstyle="-|>",
            color=(0, 0, 0),
            alpha=line_alpha,
            zorder=-100000)

        ax.add_artist(a)

    if traces is not None:
        for interpolated_positions, interpolated_times, cols in traces:

            ax.scatter(
                interpolated_times,
                interpolated_positions[:, 0],
                interpolated_positions[:, 1],
                c=cols
            )

    ax.set_xlabel('Generation Number (Creation Time)')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    plt.title(title)

    return fig, ax


def visualize_genomes3D_GIF(
        save_fpath,
        graph: nx.DiGraph,
        positions: dict,
        genome_colors: dict,
        traces: list=None,
        dimmer_list: list=None,
        **kwargs):

    fig, ax = _visualize_genomes3D(
        graph=graph,
        positions=positions,
        genome_colors=genome_colors,
        traces=traces,
        dimmer_list=dimmer_list,
        **kwargs)

    image_type = kwargs["vis_image_type"]
    increment = kwargs["increment"]
    n_seconds = kwargs["n_seconds"]

    kwargs.setdefault('angle_offset_3D', increment)

    angles = np.arange(0, 360, increment) - 180 + kwargs["angle_offset_3D"]
    viewpoints = [(30, angle) for angle in angles]

    frames = []
    for elev, azim in viewpoints:
        ax.view_init(elev=elev, azim=azim)

        buffer = io.BytesIO()
        plt.savefig(buffer, format=image_type)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))
        buffer.close()

    valid_formats = ['.gif', '.mp4', '.avi', '.mov']
    _, ext_ = os.path.splitext(save_fpath)
    ext_ = ext_.lower()

    if ext_ not in valid_formats:
        save_fpath += ".gif"

    # save the GIF
    imageio.mimsave(save_fpath, frames, fps=len(viewpoints) / n_seconds)
    print(f"GIF saved at {save_fpath}")

    plt.close(fig)


def visualize_genomes3D_images(
        save_fpath,
        graph: nx.DiGraph,
        positions: dict,
        genome_colors: dict,
        traces: list=None,
        dimmer_list: list=None,
        best_genome_id: int=None,
        **kwargs):


    fig, ax = _visualize_genomes3D(
        graph=graph,
        positions=positions,
        genome_colors=genome_colors,
        traces=traces,
        dimmer_list=dimmer_list,
        best_genome_id=best_genome_id,
        **kwargs)

    image_type = kwargs["vis_image_type"]
    num_frames = kwargs["num_frames"]

    increment = 360 / num_frames


    angles = np.arange(0, 360, increment) - 180 + kwargs["angle_offset_3D"]
    viewpoints = [(30, angle) for angle in angles]

    os.makedirs(save_fpath, exist_ok=True)

    for idx, (elev, azim) in enumerate(viewpoints):
        ax.view_init(elev=elev, azim=azim)
        combined_path = os.path.join(save_fpath, f"frame_{idx + 1}.{image_type}")
        plt.savefig(combined_path, format=image_type)

    print(f"GIF saved at {save_fpath}")

    plt.close(fig)
