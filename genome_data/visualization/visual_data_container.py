import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .coloring import _set_colors_by_fitness, _set_colors_by_group
from .arrow3d import Arrow3D
import io
import imageio.v2 as imageio


class VisualDataContainer:

    def __init__(self):
        self.genome_ids = None
        self.relations = None
        self.genome_colors = None
        self.legend_handles = None

    def init_graph_data(self, genome_ids, relations):
        """
        Set the data_storage that makes up the graph of the genomes relations.

        Args:
            genome_ids: The list of all genome IDs.
            relations: All parent-child relations between the genomes.
        """
        self.genome_ids = np.array(genome_ids, dtype=np.int64)
        self.relations = np.array(relations, dtype=np.int64)

    def make_graph(self):
        """
        Convert this data_storage to a networkx graph that is usable.

        Returns: A networkx graph of the relations between the genomes.
        """
        graph = nx.DiGraph()
        for genome_id in self.genome_ids:
            graph.add_node(genome_id)

        for id_pair in self.relations:
            if (id_pair[0] in graph.nodes) and (id_pair[1] in graph.nodes):
                graph.add_edge(id_pair[0], id_pair[1])

        return graph

    def set_colors_by_fitness(self, fitness_values, col_low, col_high):
        """
        Set the colors of the genomes based on fitness values.

        Args:
            fitness_values: A dict mapping genome ID to fitness.
            col_low: The color indicating low fitness.
            col_high: The color indicating high fitness.
        """

        self.legend_handles = [
            Patch(color=col_low, label='Low Loss'),
            Patch(color=col_high, label='High Loss'),
        ]

        self.genome_colors = _set_colors_by_fitness(
            fitness_values=fitness_values, col_low=col_low, col_high=col_high)

    def set_colors_by_group(self, genome_groups):
        """
        Set the colors of the genomes based on what group they belong to.

        Args:
            genome_groups: The group number for each genome.
        """
        self.genome_colors = _set_colors_by_group(genome_groups=genome_groups)

    def set_colors(self, genome_colors: dict):
        """
        Set the colors of all the genomes.

        Args:
            genome_colors: The color for each genome.
        """
        self.genome_colors = genome_colors

    def visualize_genomes2D(
            self,
            save_fpath: str,
            positions: dict,
            args,
            seed=42):

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        genome_ids_set = set(self.genome_ids)
        assert(genome_ids_set == set(positions.keys()))
        assert(genome_ids_set == set(self.genome_colors.keys()))

        node_size = args["node_size"]
        arrow_size = args["arrow_size"]
        line_width = args["line_width"]

        graph = self.make_graph()

        colors = [self.genome_colors[int(node)] for node in graph.nodes]

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

        if self.legend_handles is not None:
            plt.legend(handles=self.legend_handles, title="Node Colors")

        plt.savefig(save_fpath)
        print(f"Saved figure to {save_fpath}")

        ax = plt.gca()
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        return x_limits, y_limits

    def make_plot3D(
            self,
            ax,
            graph: nx.DiGraph,
            positions: dict,
            args):

        node_size = args["node_size"]
        arrow_size = args["arrow_size"]
        line_width = args["line_width"]

        alpha = 1

        for node in graph.nodes:
            x, y, z = positions[node]

            ax.plot(
                x, y, z,
                marker='o',
                markersize=node_size,
                color=self.genome_colors[node],
                alpha=alpha)

        for edge in graph.edges:
            from_node, to_node = edge
            start_position = positions[from_node]
            end_position = positions[to_node]

            a = Arrow3D([start_position[0], end_position[0]],
                        [start_position[1], end_position[1]],
                        [start_position[2], end_position[2]],
                        mutation_scale=arrow_size,
                        lw=line_width,
                        arrowstyle="-|>",
                        color=(0, 0, 0),
                        alpha=alpha,
                        zorder=-100000)

            ax.add_artist(a)

        ax.set_xlabel('Generation Number (Creation Time)')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        plt.title('3D Graph Plot')

    def visualize_genomes3D(
            self,
            save_fpath: str,
            positions: dict,
            args):

        increment = args["increment"] if "increment" in args else 5
        n_seconds = args["n_seconds"] if "n_seconds" in args else 10

        angles = np.arange(0, 360, increment) - 180 + increment
        viewpoints = [(30, angle) for angle in angles]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        graph = self.make_graph()

        self.make_plot3D(
            ax=ax,
            graph=graph,
            positions=positions,
            args=args)

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
