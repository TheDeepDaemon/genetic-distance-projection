from .. import GenomeDataCollector
from .visualization import (
    visualize_genomes2D, calc_colors_by_fitness, calc_colors_by_group, trace_gene_origin,
    visualize_genomes3D_GIF, visualize_genomes3D_images)
from .genome_microscope import GenomeMicroscope
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from ..program_arguments import ProgramArguments
import torch
from scipy.interpolate import CubicSpline
from matplotlib.colors import Normalize


class GenomeVisualizer:

    def __init__(self, genome_data_collector: GenomeDataCollector):
        self.genome_data_collector = genome_data_collector
        self.genome_colors = None
        self.legend_handles = None
        self.dimmer_list = []
        self.nn_model = None

    def set_colors_by_fitness(self, fitness_values, col_low, col_high):
        """
        Set the colors of the genomes based on fitness values.

        Args:
            fitness_values: A dict mapping genome ID to fitness.
            col_low: The color indicating low fitness.
            col_high: The color indicating high fitness.
        """

        self.genome_colors, self.legend_handles = calc_colors_by_fitness(
            fitness_values=fitness_values, col_low=col_low, col_high=col_high)

    def set_genome_colors_by_group(self, genome_groups):
        """
        Set the colors of the genomes based on what group they belong to.

        Args:
            genome_groups: The group number for each genome.
        """
        self.genome_colors, self.legend_handles = calc_colors_by_group(genome_groups=genome_groups)

    def set_group_focus(self, group_number, ingroup_alpha, outgroup_alpha):
        """
        Turn the alpha down for everything outside of this group, to increase the focus on it.

        Args:
            group_number: The group to focus on.
            ingroup_alpha: The alpha to use for that group.
            outgroup_alpha: The alpha to use for other groups.
        """
        group_numbers = self.genome_data_collector.get_genome_attribute_by_key("group")
        for gid in self.genome_colors:
            if group_numbers[gid] != group_number:
                self.dimmer_list.append(gid)

    def visualize_genomes2D(
            self,
            save_fpath: str,
            args):
        """
        Perform 2D visualizations, save to an image file.

        Args:
            save_fpath: The filepath to save it to.
            args: Program arguments and their keywords.
        """

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        visualize_genomes2D(
            save_fpath=save_fpath,
            genome_data_collector=self.genome_data_collector,
            genome_colors=self.genome_colors,
            args=args,
            legend_handles=self.legend_handles)

    def predict_position_with_model(self, data: torch.Tensor | np.ndarray):
        if self.nn_model is not None:

            self.nn_model.eval()

            if isinstance(data, np.ndarray):
                data = torch.from_numpy(np.copy(data.astype(np.float32)))

                outputs = self.nn_model(data)

                return outputs.detach().cpu().numpy()
            else:
                return self.nn_model(data)

    def _interpolate_nn(self, gene_vectors, times, cmap, args):
        if gene_vectors is None or self.nn_model is None:
            return

        n_steps_between = args.interp_factor
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

        predicted_positions = self.predict_position_with_model(interpolated_genomes)

        if isinstance(predicted_positions, torch.Tensor):
            predicted_positions = predicted_positions.detach().cpu().numpy()

        norm = Normalize(vmin=0, vmax=len(predicted_positions))
        cols = cmap(norm(np.arange(len(predicted_positions))))

        return predicted_positions, interpolated_times, cols

    def _interpolate_cubic(self, points, times, cmap, args):

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

        t_interp = np.linspace(t.min(), t.max(), len(points) * args.interp_factor)
        x_interp = cs_x(t_interp)
        y_interp = cs_y(t_interp)
        interpolated_times = cs_times(t_interp)

        norm = Normalize(vmin=t_interp.min(), vmax=t_interp.max())

        cols = cmap(norm(t_interp))

        interpolated_positions = [x_interp, y_interp]
        interpolated_positions = np.array(interpolated_positions).T

        return interpolated_positions, interpolated_times, cols

    def _create_traces(self, args):

        paths = []

        if args.trace_best or args.trace_gene_origins:
            self.dimmer_list = [gid for gid in self.genome_colors]

        if args.trace_best:
            line_of_succession = self.genome_data_collector.get_line_of_succession()

            if args.interpolation_type == 'nn':
                gene_vectors = self.genome_data_collector.get_genome_attribute_by_key("genes")

                best_gene_vectors = [gene_vectors[gid] for gid in line_of_succession]
                best_gene_vectors = np.array(best_gene_vectors)

                trace = self._interpolate_nn(best_gene_vectors, line_of_succession, plt.cm.spring, args)

            elif args.interpolation_type == 'cubic':
                positions = self.genome_data_collector.get_genome_attribute_by_key("reduced_position")

                points = [positions[gid] for gid in line_of_succession]
                points = np.array(points)

                trace = self._interpolate_cubic(points, line_of_succession, plt.cm.spring, args)

            paths.append(trace)

        if args.trace_gene_origins:
            global_best = self.genome_data_collector.get_global_best()

            genes_dict = self.genome_data_collector.get_genome_attribute_by_key("edges", "n")
            parents_dict = self.genome_data_collector.get_genome_attribute_by_key("parents")

            cmaps = list(colormaps)

            for i, gene in enumerate(genes_dict[global_best]):

                gene_path = trace_gene_origin(
                    genes_dict=genes_dict,
                    parents_dict=parents_dict,
                    genome_id=global_best,
                    gene=gene)

                cmap = plt.get_cmap(cmaps[i])

                if args.interpolation_type == 'nn':
                    gene_vectors = self.genome_data_collector.get_genome_attribute_by_key("genes")

                    best_gene_vectors = [gene_vectors[gid] for gid in gene_path]
                    best_gene_vectors = np.array(best_gene_vectors)

                    trace = self._interpolate_nn(best_gene_vectors, gene_path, cmap, args)

                elif args.interpolation_type == 'cubic':
                    positions = self.genome_data_collector.get_genome_attribute_by_key("reduced_position")

                    points = [positions[gid] for gid in gene_path]
                    points = np.array(points)

                    trace = self._interpolate_cubic(points, gene_path, cmap, args)

                paths.append(trace)

        if len(paths) == 0:
            paths = None

        return paths

    def visualize_genomes3D_GIF(
            self,
            save_fpath: str,
            args: ProgramArguments,
            trace_best=False,
            trace_gene_origins=False,
            title: str=None):
        """
        Perform the 3D visualization, save to a GIF.

        Args:
            save_fpath: The filepath to save it to.
            args: Program arguments and their keywords.
            trace_best: Whether to trace a path showing the global best.
            trace_gene_origins: Whether to show the origins of the global best genes.
            title: The plot title.
        """

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        traces = self._create_traces(
            trace_best=trace_best,
            trace_gene_origins=trace_gene_origins)

        visualize_genomes3D_GIF(
            save_fpath=save_fpath,
            genome_data_collector=self.genome_data_collector,
            genome_colors=self.genome_colors,
            args=args,
            dimmer_list=self.dimmer_list,
            traces=traces,
            title=title)

    def visualize_genomes3D_images(
            self,
            save_fpath: str,
            args: ProgramArguments,
            title: str=None):
        """
        Perform the 3D visualization, save to a GIF.

        Args:
            save_fpath: The filepath to save it to.
            args: Program arguments and their keywords.
            title: The plot title.
        """

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        traces = self._create_traces(args=args)

        visualize_genomes3D_images(
            save_fpath=save_fpath,
            genome_data_collector=self.genome_data_collector,
            genome_colors=self.genome_colors,
            args=args,
            dimmer_list=self.dimmer_list,
            traces=traces,
            title=title)

    def visualize_genomes_microscope(self, args, data_collector: GenomeDataCollector):
        GenomeMicroscope(
            args=args,
            genome_data_collector=data_collector,
            genome_colors=self.genome_colors)
