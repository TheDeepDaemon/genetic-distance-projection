from gdp import GenomeVisualizer, GenomeData
import os
from datetime import datetime


def query_data_saves(data_source, reduction_type, use_gene_data, use_weight_data, truncate_to):

    # set the arguments used to identify it
    identifying_args = {
        "data_source": data_source,
        "reduction_type": reduction_type,
        "use_gene_data": use_gene_data,
        "use_weight_data": use_weight_data,
        "truncate_to": truncate_to
    }

    # doing this to demonstrate how we find the latest save and load it
    saved_data_path = GenomeData.find_latest_genome_data(
        "reduced_data",
        identifying_args=identifying_args)

    return saved_data_path


def make_title(data_source, reduction_type, use_gene_data, use_weight_data):
    title, _ = os.path.splitext(data_source)

    title = title.replace("_", " ")

    title = ' '.join(word if word.isupper() else word.capitalize() for word in title.split())

    #title += " - " + reduction_type.upper()

    if use_gene_data and use_weight_data:
        title += " - Gene Data & Weight Data"
    elif use_gene_data:
        title += " - Gene Data"
    elif use_weight_data:
        title += " - Weight Data"

    return title


def visualize_reduced_data(data_source, reduction_type, use_gene_data, use_weight_data, truncate_to):

    saved_data_path = query_data_saves(
        data_source, reduction_type, use_gene_data, use_weight_data, truncate_to)

    print(f"Loading genome data from: {saved_data_path}")

    # visualize
    genome_visualizer = GenomeVisualizer(source=saved_data_path)

    genome_visualizer.set_genome_colors_by_group()

    # make title...
    title = make_title(data_source, reduction_type, use_gene_data, use_weight_data)

    # output visualization
    os.makedirs("output", exist_ok=True)
    genome_visualizer.visualize_genomes3D_GIF(
        save_fpath=f"output/{os.path.splitext(data_source)[0]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}",
        title=title)
