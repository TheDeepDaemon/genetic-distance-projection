from ..genome_data import GenomeData
from .visualization import visualize_genomes2D, visualize_genomes3D, calc_colors_by_fitness, calc_colors_by_group
from .genome_microscope import GenomeMicroscope


class GenomeVisualizer:

    def __init__(self, genome_data: GenomeData):
        self.genome_data = genome_data
        self.genome_colors = None
        self.legend_handles = None

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
            args: dict):
        """
        Perform the 3D visualization, save to a GIF.

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

        visualize_genomes3D(
            save_fpath=save_fpath, genome_data=self.genome_data, genome_colors=self.genome_colors, args=args)

    def visualize_genomes_microscope(self, args, genome_data_dict):
        GenomeMicroscope(
            args=args,
            genome_data=self.genome_data,
            genome_colors=self.genome_colors,
            genome_data_dict=genome_data_dict)
