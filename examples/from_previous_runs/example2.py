"""examples/from_previous_runs/example2.py
This file contains an example of GDP being used to generate 2D visuals from some EXAMM data.
This is an example like the one in "example1.py" but it shows how data is saved and loaded between steps.
"""

"""
 ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
This code block is just here to make is easy to import the GDP module from this particular file.
It is not a part of the example, so you can ignore it.
Note: I know this is a hacky way to do it, but if you are just trying to run an example script,
it requires minimal input on your part.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
"""
 ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
"""
from gdp import GenomeDataCollector, ReducedGenomeData, GenomeVisualizer, reduce_using_pca
import os


def main(data_path, use_gene_data, use_weight_data):
    """
    The second example of GDP being used with a neuroevolution algorithm.

    Args:
        data_path: The path to retrieve the data from.
        use_gene_data: Whether to include gene data.
        use_weight_data: Whether to include weight data.
    """

    genome_data_collector = GenomeDataCollector.load(data_path)
    print(f"Genome data loaded from {data_path}")

    if use_gene_data:

        genome_data_collector.convert_info_to_genes(
            func=lambda data_entry: ["node_id:{}".format(gn["n"]) for gn in data_entry],
            key="nodes")

        genome_data_collector.convert_info_to_genes(
            func=lambda data_entry: ["edge_id:{}".format(ge["n"]) for ge in data_entry],
            key="edges")

        genome_data_collector.convert_info_to_genes(
            func=lambda data_entry: ["recurrent_edge:{}".format(gre["n"]) for gre in data_entry],
            key="recurrent_edges")

    if use_weight_data:

        genome_data_collector.convert_info_to_gene_values(
            func=lambda data_entry: {"edge_weight:{}".format(ge["n"]): float(ge["weight"]) for ge in data_entry},
            key="edges")

        genome_data_collector.convert_info_to_gene_values(
            func=lambda data_entry: {"recurrent_edge_weight:{}".format(gre["n"]): float(gre["weight"]) for gre in data_entry},
            key="recurrent_edges")

    os.makedirs("collected_data", exist_ok=True)
    genome_data_collector.save("collected_data/example_collected_data.zip")

    # dimensionality reduction
    reduced_genome_data = ReducedGenomeData.perform_reduction(
        source="collected_data/example_collected_data.zip",
        dim_reduction_function=reduce_using_pca)

    os.makedirs("reduced_data", exist_ok=True)
    reduced_genome_data.save("reduced_data/example_reduced_data.zip")

    # visualize
    genome_visualizer = GenomeVisualizer(source="reduced_data/example_reduced_data.zip")

    genome_visualizer.set_genome_colors_by_group()

    os.makedirs("output", exist_ok=True)
    genome_visualizer.visualize_genomes2D(save_fpath="output/example_visuals")


if __name__=="__main__":
    data_directory = "examm-neat-data"
    data_source_json = "no_enabled_info_repop_1epoch.zip"
    data_path = os.path.join(data_directory, data_source_json)
    main(data_path=data_path, use_gene_data=True, use_weight_data=False)
