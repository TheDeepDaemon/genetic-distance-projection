import numpy as np

from .. import GenomeDataCollector
from ..genome_data import GenomeData
from .visualization import (
    visualize_genomes2D, visualize_genomes3D, calc_colors_by_fitness, calc_colors_by_group, trace_gene_origin)
from .genome_microscope import GenomeMicroscope
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_rgb


class GenomeVisualizer:

    def __init__(self, genome_data: GenomeData, genome_data_collector: GenomeDataCollector):
        self.genome_data = genome_data
        self.genome_data_collector = genome_data_collector
        self.genome_colors = None
        self.legend_handles = None
        self.dimmer_list = []

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

        if self.genome_data.reduction_type_used is None:
            print("You must reduce the genomes to positions before displaying.")
            return

        visualize_genomes2D(
            save_fpath=save_fpath,
            genome_data=self.genome_data,
            genome_colors=self.genome_colors,
            args=args,
            legend_handles=self.legend_handles)

    def visualize_genomes3D(
            self,
            save_fpath: str,
            args: dict,
            trace_best=False,
            trace_gene_origins=False):
        """
        Perform the 3D visualization, save to a GIF.

        Args:
            save_fpath: The filepath to save it to.
            args: Program arguments and their keywords.
            trace_best: Whether to trace a path showing the global best.
            trace_gene_origins: Whether to show the origins of the global best genes.
        """

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        if self.genome_data.reduction_type_used is None:
            print("You must reduce the genomes to positions before displaying.")
            return

        paths = []

        if trace_best or trace_gene_origins:
            positions_dict = self.genome_data.get_positions_with_gid()
            self.dimmer_list = [gid for gid in self.genome_colors]

        if trace_best:
            line_of_succession = self.genome_data_collector.get_line_of_succession()

            points = [positions_dict[gid] for gid in line_of_succession]
            points = np.array(points)

            paths.append((points, plt.cm.spring))

        if trace_gene_origins:
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

                print(gene_path)

                cmap = plt.get_cmap(cmaps[i])

                points = [positions_dict[gid] for gid in gene_path]
                points = np.array(points)

                paths.append((points, cmap))

        if len(paths) == 0:
            paths = None

        visualize_genomes3D(
            save_fpath=save_fpath,
            genome_data=self.genome_data,
            genome_colors=self.genome_colors,
            args=args,
            dimmer_list=self.dimmer_list,
            paths_to_trace=paths)

    def visualize_genomes_microscope(self, args, data_collector: GenomeDataCollector):
        GenomeMicroscope(
            args=args,
            genome_data=self.genome_data,
            genome_data_collector=data_collector,
            genome_colors=self.genome_colors)
