import numpy as np
import matplotlib.pyplot as plt
from .arrow3d import Arrow3D
import io
import imageio.v2 as imageio
from ...genome_data import GenomeData
from .interp_3d_path import disp_interpolated_points
from matplotlib.colors import to_rgb


def visualize_genomes3D(
        save_fpath,
        genome_data: GenomeData,
        genome_colors,
        args,
        special_group=None,
        paths_to_trace=None,
        dimmer_list=None):
    """
    Perform the 3D visualization, save to a GIF.

    Args:
        save_fpath: The filepath to save the GIF to.
        genome_data: The genome data to use.
        genome_colors: The node colors for the genomes.
        args: Passed arguments and their keywords.
    """

    # MAKE THE PLOT

    # create the figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # program arguments
    node_size = args["node_size"]
    arrow_size = args["arrow_size"]
    line_width = args["line_width"]
    line_alpha = args["3D_line_alpha"]

    # function arguments
    graph = genome_data.make_graph()
    positions = genome_data.get_positions_with_gid()

    for node in graph.nodes:
        x, y, z = positions[node]

        alpha = args["3D_node_alpha_on"]

        if node in dimmer_list:
            alpha = args["3D_node_alpha_off"]

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
    plt.title('3D Graph Plot')

    # SAVE IMAGES OF THE PLOT
    increment = args["increment"] if "increment" in args else 5
    n_seconds = args["n_seconds"] if "n_seconds" in args else 10

    angles = np.arange(0, 360, increment) - 180 + increment
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
