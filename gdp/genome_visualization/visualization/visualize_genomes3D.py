import numpy as np
import matplotlib.pyplot as plt
from .arrow3d import Arrow3D
import io
import imageio.v2 as imageio
from ...genome_data import GenomeDataCollector
from .interp_3d_path import disp_interpolated_points
import os
from ...program_arguments import ProgramArguments


def _visualize_genomes3D(
        genome_data_collector: GenomeDataCollector,
        genome_colors,
        args: ProgramArguments,
        paths_to_trace: list=None,
        dimmer_list: list=None,
        title: str=None):
    """
    Perform the 3D visualization, save to a GIF.

    Args:
        genome_data_collector: The genome data.
        genome_colors: The node colors for the genomes.
        args: Passed arguments and their keywords.
        paths_to_trace: The list of paths to trace, and their color maps.
        dimmer_list: A list of nodes to dim.
        title: The title to show in the 3D graph plot.
    """

    # create the figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # program arguments
    node_size = args.node_size
    arrow_size = args.arrow_size
    line_width = args.line_width
    line_alpha = args.line_alpha_3D

    # function arguments
    graph = genome_data_collector.make_graph()
    positions = genome_data_collector.get_3D_positions()

    for node in graph.nodes:
        x, y, z = positions[node]

        alpha = args.node_alpha_on_3D

        if node in dimmer_list:
            alpha = args.node_alpha_off_3D

        ax.plot(
            x, y, z,
            marker='o',
            markersize=node_size,
            color=genome_colors[node],
            alpha=alpha)

    for edge in graph.edges:
        from_node, to_node = edge
        start_position = positions[from_node]
        end_position = positions[to_node]

        a = Arrow3D(
            [start_position[0], end_position[0]],
            [start_position[1], end_position[1]],
            [start_position[2], end_position[2]],
            mutation_scale=arrow_size,
            lw=line_width,
            arrowstyle="-|>",
            color=(0, 0, 0),
            alpha=line_alpha,
            zorder=-100000)

        ax.add_artist(a)

    if paths_to_trace is not None:
        for path, cmap in paths_to_trace:
            disp_interpolated_points(ax=ax, points=path, cmap=cmap)

    ax.set_xlabel('Generation Number (Creation Time)')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    if title is not None:
        plt.title(title)
    else:
        plt.title('3D Graph Plot')

    return fig, ax


def visualize_genomes3D_GIF(
        save_fpath,
        genome_data_collector: GenomeDataCollector,
        genome_colors,
        args: ProgramArguments,
        paths_to_trace: list=None,
        dimmer_list: list=None,
        title: str=None):

    fig, ax = _visualize_genomes3D(
        genome_data_collector=genome_data_collector,
        genome_colors=genome_colors,
        args=args,
        paths_to_trace=paths_to_trace,
        dimmer_list=dimmer_list,
        title=title)

    increment = args.increment
    n_seconds = args.n_seconds

    angles = np.arange(0, 360, increment) - 180 + args.angle_offset_3D
    viewpoints = [(30, angle) for angle in angles]

    frames = []
    for elev, azim in viewpoints:
        ax.view_init(elev=elev, azim=azim)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        frames.append(imageio.imread(buffer))
        buffer.close()

    # save the GIF
    imageio.mimsave(save_fpath, frames, fps=len(viewpoints) / n_seconds)
    print(f"GIF saved at {save_fpath}")

    plt.close(fig)


def visualize_genomes3D_images(
        save_fpath,
        genome_data_collector: GenomeDataCollector,
        genome_colors,
        args: ProgramArguments,
        paths_to_trace: list=None,
        dimmer_list: list=None,
        title: str=None):

    fig, ax = _visualize_genomes3D(
        genome_data_collector=genome_data_collector,
        genome_colors=genome_colors,
        args=args,
        paths_to_trace=paths_to_trace,
        dimmer_list=dimmer_list,
        title=title)

    num_frames = args.num_frames

    image_type = args.vis_image_type

    increment = 360 / num_frames

    angles = np.arange(0, 360, increment) - 180 + args.angle_offset_3D
    viewpoints = [(30, angle) for angle in angles]

    os.makedirs(save_fpath, exist_ok=True)

    for idx, (elev, azim) in enumerate(viewpoints):
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(os.path.join(save_fpath, f"frame_{idx + 1}.{args.vis_image_type}"), format=image_type)

    print(f"GIF saved at {save_fpath}")

    plt.close(fig)
