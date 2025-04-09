from ...genome_data import GenomeDataCollector
from ...program_arguments import ProgramArguments
import numpy as np
import matplotlib.pyplot as plt
from .arrow3d import Arrow3D
import io
import os
import imageio.v2 as imageio
from .transform_01 import apply_transformation_to01


def _visualize_genomes3D(
        genome_data_collector: GenomeDataCollector,
        genome_colors,
        args: ProgramArguments,
        traces: list=None,
        dimmer_list: list=None,
        title: str=None):
    """
    Perform the 3D visualization, save to a GIF.

    Args:
        genome_data_collector: The genome data.
        genome_colors: The node colors for the genomes.
        args: Passed arguments and their keywords.
        traces: The list of paths to trace, and their color maps.
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

    positions = genome_data_collector.get_genome_attribute_by_key("reduced_position")

    if args.transform_to_01:
        rotation_mat, offset = genome_data_collector.get_rotation_to01()
        positions = apply_transformation_to01(positions=positions, rotation_mat=rotation_mat, offset=offset)

    for node_gid in graph.nodes:
        x, y = positions[node_gid]

        alpha = args.node_alpha_on_3D

        if node_gid in dimmer_list:
            alpha = args.node_alpha_off_3D

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

        a = Arrow3D(
            [from_node_gid, to_node_gid],
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

            if args.transform_to_01:
                interpolated_positions -= offset
                interpolated_positions = np.dot(interpolated_positions, rotation_mat)

            ax.scatter(
                interpolated_times,
                interpolated_positions[:, 0],
                interpolated_positions[:, 1],
                c=cols
            )

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
        traces: list=None,
        dimmer_list: list=None,
        title: str=None):

    fig, ax = _visualize_genomes3D(
        genome_data_collector=genome_data_collector,
        genome_colors=genome_colors,
        args=args,
        traces=traces,
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
        traces: list=None,
        dimmer_list: list=None,
        title: str=None):

    fig, ax = _visualize_genomes3D(
        genome_data_collector=genome_data_collector,
        genome_colors=genome_colors,
        args=args,
        traces=traces,
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
