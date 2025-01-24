import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from typing import List, Optional, Tuple
import numpy as np
import json
from gene_data_processing import dict_values_to_percentiles, to_color, reduce_genome
from matplotlib.patches import Patch
import argparse


def create_and_save_figure(
        graph: nx.DiGraph,
        positions: dict,
        node_size: int|float,
        arrow_size: int|float,
        width: int|float,
        format: str,
        node_colors: dict,
        fpath: str,
        legend_handles=None,
        x_limits=None,
        y_limits=None
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Display a graph with positions and node colors.

    Args:
        graph (nx.DiGraph): The graph of relations.
        positions (dict): The positions of the nodes.
        temporal_data (List[int]): The genomes at each timestep.
        node_size (int or float): The size of the nodes.
        node_colors (dict): The colord of the nodes.
        base_fname (str): The basic name of the file for the figure to be saved to.
        directory (str): The directory being used for this run.
    """

    # create the layout for the graph
    pos = nx.spring_layout(graph, pos=positions, fixed=positions.keys(), seed=42)

    if isinstance(node_colors, dict):
        # convert colors to lists for nodes and outlines
        colors = [node_colors[node] for node in graph.nodes]
        #outlines = [outline_colors[node] for node in graph.nodes]
    else:
        colors = node_colors

    plt.figure(figsize=(6, 6))

    if (x_limits is not None) and (y_limits is not None):
        plt.xlim(*x_limits)
        plt.ylim(*y_limits)

    # draw nodes with fill color on top
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_color=colors,
        node_size=node_size,
        arrows=True,
        arrowsize=arrow_size,
        width=width)

    if legend_handles is not None:
        plt.legend(handles=legend_handles, title="Node Categories")

    fname = f"{fpath}.{format}"
    plt.savefig(fname)
    print(f"Saved figure to {fname}")

    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    return x_limits, y_limits


def save_log(log_data, fpath: str):
    """
    Save the figure using matplotlib.pyplot.

    Args:
        log_data (dict): A dict with all of the log data we want to save.
        fpath (str): The file path to save it to.
    """

    fpath = f"{fpath}.txt"

    # save it, log path
    with open(fpath, 'w') as f:
        f.write(json.dumps(log_data))
        print(f"Saved log to {fpath}")


def parse_temporal_data(temporal_data, positions):
    positions_over_time = []
    for genome_list in temporal_data:
        positions_list = []
        for genome_id in genome_list:
            positions_list.append(positions[genome_id])

        positions_arr = np.array(positions_list)
        positions_over_time.append(positions_arr)

    return positions_over_time


def show_positions_over_time(temporal_data, positions, colors, x_limits, y_limits, fpath):
    """
    Display positions over time as an animation based on temporal data and save it as a video file.

    Args:
        temporal_data (list of list of int): Each inner list contains IDs representing object positions at each time step.
        positions (dict): Maps an ID to an (x, y) position.
        colors (dict): Maps an ID to an RGB color tuple (each value in the range [0, 1]).
        x_limits (tuple): (x_min, x_max) limits for the plot.
        y_limits (tuple): (y_min, y_max) limits for the plot.
        fpath (str): File path to save the animation (without extension).
    """
    fig, ax = plt.subplots()
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    # Create a scatter plot for each object based on the number of unique IDs in temporal_data
    scatters = [ax.plot([], [], 'o')[0] for _ in range(len(temporal_data[0]))]

    # Initialize scatter plot positions and colors
    for idx, scatter in enumerate(scatters):
        scatter.set_data([], [])
        obj_id = temporal_data[0][idx]
        if obj_id in colors:
            scatter.set_color(colors[obj_id])

    # Update scatter plot for each frame
    def animate(frame):
        if frame < len(temporal_data):
            ids_at_frame = temporal_data[frame]
            for idx, obj_id in enumerate(ids_at_frame):
                if obj_id in positions:
                    x, y = positions[obj_id]
                    scatters[idx].set_data([x], [y])  # Wrap x and y in lists
                    if obj_id in colors:
                        scatters[idx].set_color(colors[obj_id])  # Set color
        return scatters

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=len(temporal_data), interval=500, blit=True)

    # Save the animation
    ani.save(f"{fpath}.mp4", writer="ffmpeg", dpi=100)


def create_graph(graph_nodes, graph_edges):
    graph = nx.DiGraph()
    for node in graph_nodes:
        graph.add_node(node)

    for edge in graph_edges:
        graph.add_edge(edge[0], edge[1])

    return graph


def decode_dict(d):
    return {int(k): v for k, v in d.items()}


def run_visualizations(input_fname, output_fname, color_type, node_size, arrow_size, width, format):

    # output to json file
    with open(input_fname, 'r', encoding="utf-8") as file:
        contents = json.load(file)
        graph_nodes = contents['graph_nodes']
        graph_edges = contents['graph_edges']
        positions = decode_dict(contents['positions'])
        positions = {k: np.array(v) for k, v in positions.items()}
        genes_mat = np.array(contents['genes_mat'])
        genome_id_to_index = decode_dict(contents['genome_id_to_index'])
        genome_depths = decode_dict(contents['genome_depths'])
        tree_depth = contents['tree_depth']
        fitnesses = decode_dict(contents['fitnesses'])

    graph = create_graph(graph_nodes=graph_nodes, graph_edges=graph_edges)

    node_colors = None
    legend_handles = None

    if color_type=='fitness':

        # use fitness to determine colors
        red = (1.0, 0.0, 0.0)
        blue = (0.0, 0.0, 1.0)
        col_low = red
        col_high = blue
        fitness_colors = dict_values_to_percentiles(fitnesses=fitnesses)
        node_colors = {gid: to_color(f, col_low=col_low, col_high=col_high) for gid, f in fitness_colors.items()}

        legend_handles = [
            Patch(color=col_low, label='Low Loss'),
            Patch(color=col_high, label='High Loss'),
        ]

    elif color_type=='pca':

        # use PCA to determine colors
        node_colors = reduce_genome(
            genes_matrix=genes_mat,
            genome_id_to_index=genome_id_to_index,
            reduction_method='pca',
            reduced_size=3,
            reduction_type='color')

    elif color_type=='timing':

        # use timings to determine colors
        timing_colors = {gid: (t / tree_depth) for gid, t in genome_depths.items()}
        node_colors = {gid: (b, b, b) for gid, b in timing_colors.items()}

    elif color_type=='d-fitness':

        node_colors = {gid: (0, 0, 1) for gid in genome_id_to_index}

        for gid, idx in genome_id_to_index.items():
            parents = list(graph.predecessors(gid))

            if len(parents) > 0:
                parent_fitnesses = np.array([fitnesses[p] for p in parents])

                p_fitness = np.average(parent_fitnesses)

                if fitnesses[gid] < p_fitness:
                    node_colors[gid] = (1, 0, 0)


    create_and_save_figure(
        graph=graph,
        positions=positions,
        node_size=node_size,
        arrow_size=arrow_size,
        width=width,
        format=format,
        node_colors=node_colors,
        fpath=output_fname,
        legend_handles=legend_handles)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run visualizations with specified parameters.")

    # Add arguments
    parser.add_argument(
        "input_fname",
        type=str,
        help="Filename of the input file containing data for visualization."
    )
    parser.add_argument(
        "output_fname",
        type=str,
        help="Filename for saving the visualization output."
    )
    parser.add_argument(
        "--color_type",
        type=str,
        choices=['fitness', 'pca', 'timing', 'd-fitness'],
        default='fitness',
        help="Type of color to use in the visualization."
    )
    parser.add_argument(
        "--node_size",
        type=int,
        default=40,
        help="Size of the nodes in the visualization."
    )
    parser.add_argument(
        "--arrow_size",
        type=int,
        default=2,
        help="Size of the arrows in the visualization."
    )
    parser.add_argument(
        "--width",
        type=float,
        default=1,
        help="Size of the nodes in the visualization."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['png', 'svg'],
        default='png',
        help="Format to save the figure with.")

    # Parse arguments
    args = parser.parse_args()

    # Pass arguments to the `run_visualizations` function
    run_visualizations(
        input_fname=args.input_fname,
        output_fname=args.output_fname,
        color_type=args.color_type,
        node_size=args.node_size,
        arrow_size=args.arrow_size,
        width=args.width,
        format=args.format)
