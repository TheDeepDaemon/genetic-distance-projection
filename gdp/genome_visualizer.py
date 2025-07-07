from .reduced_genome_data import ReducedGenomeData
from ._visualization_ import (
    visualize_genomes2D, calc_colors_by_fitness, calc_colors_by_group, trace_gene_origin,
    visualize_genomes3D_GIF, visualize_genomes3D_images)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
import networkx as nx
import torch
from typing import Union
import os
from .__config__ import _set_kwargs_defaults_


class GenomeVisualizer(ReducedGenomeData):

    def __init__(
            self,
            source: Union[ReducedGenomeData, str, os.PathLike]=None,
            *args,
            **kwargs):
        if isinstance(source, (ReducedGenomeData, str, os.PathLike)):
            super().__init__(source=source, *args, **kwargs)
        else:
            raise TypeError("Expected GenomeData instance or path to file.")

        self.genome_colors = None
        self.legend_handles = None
        self.dimmer_list = []
        self.nn_model = None

    @staticmethod
    def load(zip_fpath: Union[str, os.PathLike]):
        genome_visualizer: GenomeVisualizer = GenomeVisualizer()
        genome_visualizer._load_from_path(zip_fpath=zip_fpath)
        return genome_visualizer

    def make_graph(self):
        """
        Convert this data_storage to a networkx graph that is usable.

        Returns:
            A networkx graph of the relations between the genomes.
        """
        graph = nx.DiGraph()
        for genome_id in self._population_info:
            graph.add_node(genome_id)

        for gid, info in self._population_info.items():
            parents = info["parents"]
            for pid in parents:
                if pid != gid:
                    if (pid in graph.nodes) and (gid in graph.nodes):
                        graph.add_edge(pid, gid)

        return graph

    def get_single_attribute(self, genome_id, key):
        return self._population_info[genome_id][key]

    def get_genome_attribute_by_key(self, *args):
        if len(args) == 1:
            key = args[0]
            return {genome_id: info[key] for genome_id, info in self._population_info.items()}
        elif len(args) == 2:
            outer_key, inner_key = args
            result = dict()
            for genome_id, info in self._population_info.items():
                inner_list = []
                for item in info[outer_key]:
                    inner_list.append(item[inner_key])

                result[genome_id] = inner_list
            return result
        else:
            raise TypeError(f"get_genome_attribute_by_key expected 1 or 2 arguments, got {len(args)}")

    def get_line_of_succession(self, fitness_key="fitness"):
        line_of_succession = []

        genome_id_list = self.get_unique_genome_id_list()
        genome_id_list.sort()

        fitnesses = self.get_genome_attribute_by_key(fitness_key)

        if self._minimizing_fitness:
            current_best_fitness = float('inf')
        else:
            current_best_fitness = float('-inf')

        for genome_id in genome_id_list:
            gf = fitnesses[genome_id]
            if self._minimizing_fitness:
                if gf < current_best_fitness:
                    current_best_fitness = gf
                    line_of_succession.append(genome_id)
            else:
                if gf > current_best_fitness:
                    current_best_fitness = gf
                    line_of_succession.append(genome_id)

        return line_of_succession

    def get_global_best(self, fitness_key="fitness"):

        genome_id_list = self.get_unique_genome_id_list()
        genome_id_list.sort()

        fitnesses = self.get_genome_attribute_by_key(fitness_key)

        if self._minimizing_fitness:
            current_best_fitness = float('inf')
        else:
            current_best_fitness = float('-inf')

        current_best_genome = None

        for genome_id in genome_id_list:
            gf = fitnesses[genome_id]
            if self._minimizing_fitness:
                if gf < current_best_fitness:
                    current_best_fitness = gf
                    current_best_genome = genome_id
            else:
                if gf > current_best_fitness:
                    current_best_fitness = gf
                    current_best_genome = genome_id

        return current_best_genome

    def set_colors_by_fitness(self, col_low, col_high, fitness_key: str="fitness"):
        """
        Set the colors of the genomes based on fitness values.

        Args:
            fitness_key: The name of the fitness attribute.
            col_low: The color indicating low fitness.
            col_high: The color indicating high fitness.
        """
        fitness_values = self.get_genome_attribute_by_key(fitness_key)
        self.genome_colors, self.legend_handles = calc_colors_by_fitness(
            fitness_values=fitness_values, col_low=col_low, col_high=col_high)

    def set_genome_colors_by_group(self, group_key: str="group"):
        """
        Set the colors of the genomes based on what group they belong to.

        Args:
            group_key: The name of the group attribute.
        """
        genome_groups = self.get_genome_attribute_by_key(group_key)
        self.genome_colors, self.legend_handles = calc_colors_by_group(genome_groups=genome_groups)

    def set_genome_colors(self, genome_colors: dict):
        self.genome_colors = {int(gid): col for gid, col in genome_colors.items()}

    def set_group_focus(self, group_number):
        """
        Turn the alpha down for everything outside of this group, to increase the focus on it.

        Args:
            group_number: The group to focus on.
        """
        group_numbers = self.get_genome_attribute_by_key("group")
        for gid in self.genome_colors:
            if group_numbers[gid] != group_number:
                self.dimmer_list.append(gid)

    def _get_rotation_to_01(self, root_genome_id: int=None):

        best_genome_id = self.get_global_best()

        positions = {
            gid: np.array(pos, dtype=np.float32) for gid, pos in self.reduced_positions.items()}

        # if there is a root genome passed in, use it to center the data
        if root_genome_id is not None:

            root_genome_pos = positions[root_genome_id]

            offset = root_genome_pos

            # center it at the root genome
            positions = {gid: (pos - root_genome_pos) for gid, pos in positions.items()}
        else:
            offset = np.zeros(2, dtype=np.float32)

        # get the position of the best genome
        best_genome_pos = positions[best_genome_id]

        # get the magnitude of the position of the best genome
        best_genome_magnitude = np.linalg.norm(best_genome_pos)

        # get the angle of rotation
        theta = np.arctan2(float(best_genome_pos[1]), float(best_genome_pos[0]))

        # get the sin and cos for the rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # create the rotation matrix
        rotation_mat = np.array(
            [
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]],
            dtype=np.float32)

        # make sure it is normalized to a 0 to 1 range
        rotation_mat /= best_genome_magnitude

        return rotation_mat, offset

    @staticmethod
    def _apply_rotation_to_01(positions, rotation_mat, offset):
        positions = {gid: (pos - offset) for gid, pos in positions.items()}
        positions = {gid: np.dot(pos, rotation_mat) for gid, pos in positions.items()}

        return positions

    def _predict_position_with_model(self, data: torch.Tensor | np.ndarray):
        if self.nn_model is not None:

            self.nn_model.eval()

            if isinstance(data, np.ndarray):
                data = torch.from_numpy(np.copy(data.astype(np.float32)))

                outputs = self.nn_model(data)

                return outputs.detach().cpu().numpy()
            else:
                return self.nn_model(data)

    def _interpolate_nn(self, gene_vectors, times, cmap, **kwargs):
        if gene_vectors is None or self.nn_model is None:
            return

        n_steps_between = kwargs["interp_factor"]
        interpolated_genomes = []
        interpolated_times = []

        for i in range(len(gene_vectors) - 1):
            start = gene_vectors[i]
            end = gene_vectors[i + 1]
            for alpha in np.linspace(0, 1, n_steps_between, endpoint=False):
                interpolated_genomes.append((1 - alpha) * start + alpha * end)

            t_init = times[i]
            t_final = times[i + 1]
            for alpha in np.linspace(0, 1, n_steps_between, endpoint=False):
                interpolated_times.append((1 - alpha) * t_init + alpha * t_final)

        interpolated_genomes.append(gene_vectors[-1])
        interpolated_genomes = np.array(interpolated_genomes)

        interpolated_times.append(times[-1])
        interpolated_times = np.array(interpolated_times)

        predicted_positions = self._predict_position_with_model(interpolated_genomes)

        if isinstance(predicted_positions, torch.Tensor):
            predicted_positions = predicted_positions.detach().cpu().numpy()

        norm = Normalize(vmin=0, vmax=len(predicted_positions))
        cols = cmap(norm(np.arange(len(predicted_positions))))

        return predicted_positions, interpolated_times, cols

    @staticmethod
    def _interpolate_cubic(points, times, cmap, **kwargs):

        if points is None:
            return

        if len(points) <= 1:
            return points, times, cmap(0)

        x = points[:, 0]
        y = points[:, 1]

        t = np.arange(len(points))

        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_times = CubicSpline(t, times)

        t_interp = np.linspace(t.min(), t.max(), len(points) * kwargs["interp_factor"])
        x_interp = cs_x(t_interp)
        y_interp = cs_y(t_interp)
        interpolated_times = cs_times(t_interp)

        norm = Normalize(vmin=t_interp.min(), vmax=t_interp.max())

        cols = cmap(norm(t_interp))

        interpolated_positions = [x_interp, y_interp]
        interpolated_positions = np.array(interpolated_positions).T

        return interpolated_positions, interpolated_times, cols

    def _create_traces(self, **kwargs):

        trace_best = kwargs["trace_best"]
        trace_gene_origins = kwargs["trace_gene_origins"]
        interpolation_type = kwargs["interpolation_type"]

        paths = []

        if trace_best or trace_gene_origins:
            self.dimmer_list = [gid for gid in self.genome_colors]

        if trace_best:
            line_of_succession = self.get_line_of_succession()

            if interpolation_type == 'nn':
                gene_vectors = self.get_genome_attribute_by_key("genes")

                best_gene_vectors = [gene_vectors[gid] for gid in line_of_succession]
                best_gene_vectors = np.array(best_gene_vectors)

                trace = self._interpolate_nn(best_gene_vectors, line_of_succession, plt.cm.spring, **kwargs)

                paths.append(trace)

            elif interpolation_type == 'cubic':
                positions = self.reduced_positions

                points = [positions[gid] for gid in line_of_succession]
                points = np.array(points)

                trace = GenomeVisualizer._interpolate_cubic(points, line_of_succession, plt.cm.spring, **kwargs)

                paths.append(trace)

        if trace_gene_origins:
            global_best = self.get_global_best()

            genes_dict = self.get_genome_attribute_by_key("edges", "n")
            parents_dict = self.get_genome_attribute_by_key("parents")

            cmaps = list(colormaps)

            for i, gene in enumerate(genes_dict[global_best]):

                gene_path = trace_gene_origin(
                    genes_dict=genes_dict,
                    parents_dict=parents_dict,
                    genome_id=global_best,
                    gene=gene)

                cmap = plt.get_cmap(cmaps[i])

                if kwargs["interpolation_type"] == 'nn':
                    gene_vectors = self.get_genome_attribute_by_key("genes")

                    best_gene_vectors = [gene_vectors[gid] for gid in gene_path]
                    best_gene_vectors = np.array(best_gene_vectors)

                    trace = self._interpolate_nn(best_gene_vectors, gene_path, cmap, **kwargs)

                elif kwargs["interpolation_type"] == 'cubic':
                    positions = self.get_genome_attribute_by_key("reduced_position")

                    points = [positions[gid] for gid in gene_path]
                    points = np.array(points)

                    trace = self._interpolate_cubic(points, gene_path, cmap, **kwargs)

                paths.append(trace)

        return paths

    def visualize_genomes2D(
            self,
            save_fpath: str,
            **kwargs):

        _set_kwargs_defaults_(kwargs)

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        traces = self._create_traces(**kwargs)

        transform_to_01 = kwargs["transform_to_01"]
        if transform_to_01:
            rotation_mat, offset = self._get_rotation_to_01()
            positions = GenomeVisualizer._apply_rotation_to_01(
                self.reduced_positions, rotation_mat, offset)

            for i in range(len(traces)):
                pos, interpolated_times, cols = traces[i]
                pos = (pos - offset)
                pos = np.dot(pos, rotation_mat)
                traces[i] = pos, interpolated_times, cols
        else:
            positions = self.reduced_positions

        visualize_genomes2D(
            save_fpath=save_fpath,
            graph=self.make_graph(),
            positions=positions,
            genome_colors=self.genome_colors,
            legend_handles=self.legend_handles,
            **kwargs)

    def visualize_genomes3D_GIF(
            self,
            save_fpath: str,
            title: str=None,
            **kwargs):
        """
        Perform the 3D _visualization_, save to a GIF.

        Args:
            save_fpath: The filepath to save it to.
            title: The plot title.
        """

        _set_kwargs_defaults_(kwargs)

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        traces = self._create_traces(**kwargs)

        transform_to_01 = kwargs["transform_to_01"]
        if transform_to_01:
            rotation_mat, offset = self._get_rotation_to_01()
            positions = GenomeVisualizer._apply_rotation_to_01(
                self.reduced_positions, rotation_mat, offset)

            for i in range(len(traces)):
                pos, interpolated_times, cols = traces[i]
                pos = (pos - offset)
                pos = np.dot(pos, rotation_mat)
                traces[i] = pos, interpolated_times, cols
        else:
            positions = self.reduced_positions

        visualize_genomes3D_GIF(
            save_fpath=save_fpath,
            graph=self.make_graph(),
            positions=positions,
            genome_colors=self.genome_colors,
            dimmer_list=self.dimmer_list,
            traces=traces,
            title=title,
            **kwargs)

    def visualize_genomes3D_images(
            self,
            save_fpath: str,
            title: str=None,
            **kwargs):
        """
        Perform the 3D _visualization_, save to a GIF.

        Args:
            save_fpath: The filepath to save it to.
            title: The plot title.
        """

        _set_kwargs_defaults_(kwargs)

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        traces = self._create_traces(**kwargs)

        transform_to_01 = kwargs["transform_to_01"]
        if transform_to_01:
            rotation_mat, offset = self._get_rotation_to_01()
            positions = GenomeVisualizer._apply_rotation_to_01(
                self.reduced_positions, rotation_mat, offset)

            for i in range(len(traces)):
                pos, interpolated_times, cols = traces[i]
                pos = (pos - offset)
                pos = np.dot(pos, rotation_mat)
                traces[i] = pos, interpolated_times, cols
        else:
            positions = self.reduced_positions

        visualize_genomes3D_images(
            save_fpath=save_fpath,
            graph=self.make_graph(),
            positions=positions,
            genome_colors=self.genome_colors,
            dimmer_list=self.dimmer_list,
            traces=traces,
            title=title, **kwargs)
