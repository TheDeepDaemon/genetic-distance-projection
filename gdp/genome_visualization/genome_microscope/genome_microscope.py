from .make_nn_graph import make_nn_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from ...genome_data import GenomeData


class GenomeMicroscope:

    def __init__(
            self,
            args: dict,
            genome_data: GenomeData,
            genome_colors,
            genome_data_dict: dict):

        # init variables
        self.genome_node_size = args["node_size"]
        self.subgraph_node_size = args["subgraph_node_size"]

        # get arguments
        self.subgraph_width = args["subgraph_width"]
        self.subgraph_height = args["subgraph_height"]
        self.x_spacing = args["subgraph_x_spacing"]
        self.y_spacing = args["subgraph_y_spacing"]

        # the radius is gotten from the area
        self.genome_node_radius = (self.genome_node_size / math.pi)**(1/2)

        # ___ ___ ___ ___
        # set all genome data
        self.genome_data_dict = genome_data_dict

        # ___ ___ ___ ___
        # init the color map
        tableau_colors = list(mcolors.TABLEAU_COLORS.values())

        unique_neuron_types = {
            neuron["type"] for g in self.genome_data_dict.values() for neuron in g["nodes"]}

        indices = np.arange(len(unique_neuron_types))

        np.random.shuffle(indices)

        self.neuron_cmap = {nt: tableau_colors[i] for i, nt in zip(indices, unique_neuron_types)}

        # ___ ___ ___ ___
        # init the genome graph

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.margins(x=0.4, y=0.4)

        genome_graph = genome_data.make_graph()

        genome_ids_set = set(genome_graph.nodes)

        self.spr_layout = nx.spring_layout(genome_graph, pos=genome_data.get_positions(), fixed=genome_ids_set)

        self.genome_positions = np.array(
            [self.ax.transData.transform(v) for k, v in self.spr_layout.items()], dtype=np.float32)
        self.node_order = np.array(
            [k for k, v in self.spr_layout.items()], dtype=int)

        node_colors = [genome_colors[int(node)] for node in genome_graph.nodes]

        # draw nodes with fill color on top
        nx.draw(
            genome_graph,
            self.spr_layout,
            with_labels=False,
            node_color=node_colors,
            node_size=self.genome_node_size,
            arrows=True,
            arrowsize=2,
            width=0.2)

        self.subplot = None

        # add the on click function
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # make the legend for neurons
        legend_handles = [Patch(color=v, label=k) for k, v in self.neuron_cmap.items()]

        plt.legend(
            handles=legend_handles,
            title="Neuron Types",
            loc='center left',
            bbox_to_anchor=(-0.15, 0.5),
            prop={'size': 6})

        self.prev_selected_node = None

        # ___ ___ ___ ___
        # RUN THE PROGRAM
        plt.show()

    def update_subplot(self, new_subplot):
        if self.subplot is not None:
            self.subplot.remove()
            del self.subplot
            self.subplot = new_subplot
        else:
            self.subplot = new_subplot

    def on_click(self, event):

        if event.inaxes is not self.ax:
            return

        mouse_pos = np.array([event.x, event.y], dtype=np.float32)

        distances = np.linalg.norm(self.genome_positions - mouse_pos, axis=1)

        min_idx = np.argmin(distances)

        selected_node = self.node_order[min_idx]

        if selected_node != self.prev_selected_node:

            this_genome_data = self.genome_data_dict[selected_node]

            subgraph, positions = make_nn_graph(this_genome_data, self.x_spacing, self.y_spacing)

            sub_ax = self.fig.add_axes([0, 0, self.subgraph_width, self.subgraph_height])
            sub_ax.axis('off')

            scaling_factor = min(self.subgraph_width, self.subgraph_height)
            node_size = self.subgraph_node_size * scaling_factor

            neuron_colors = {
                gd["n"]: self.neuron_cmap[gd["type"]] for gd in this_genome_data["nodes"]}

            n_colors = [neuron_colors[nid] for nid in subgraph.nodes]

            edge_colors = {
                (edge["in"], edge["on"]): "#000000"
                for edge in this_genome_data["edges"]}

            recurrent_edge_colors = {
                (redge["in"], redge["on"]): "#1E90FF"
                for redge in this_genome_data["recurrent_edges"]}

            edge_colors.update(recurrent_edge_colors)

            e_colors = [edge_colors[eid] for eid in subgraph.edges]

            edge_labels = dict()
            for recurrent_edge in this_genome_data["recurrent_edges"]:
                edge_labels[(recurrent_edge["in"], recurrent_edge["on"])] = recurrent_edge["rd"]

            nx.draw(
                subgraph,
                positions,
                ax=sub_ax,
                with_labels=False,
                node_color=n_colors,
                edge_color=e_colors,
                node_size=node_size)

            nx.draw_networkx_edge_labels(
                subgraph,
                positions,
                ax=sub_ax,
                edge_labels=edge_labels,
                font_size=4,
                rotate=False)

            self.update_subplot(sub_ax)

            plt.draw()


