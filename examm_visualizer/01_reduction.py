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
from gdp import GenomeDataCollector, ReducedGenomeData
from gdp import reduce_using_mds, reduce_using_pca
from gdp import set_config_defaults
from datetime import datetime
import os
import yaml


def get_reduced(data_path, reduction_type, use_gene_data, use_weight_data, truncate_to):

    genome_data_collector = GenomeDataCollector.load(data_path)
    print(f"Genome data loaded from {data_path}")

    # truncating the data
    if truncate_to:

        genome_ids = genome_data_collector.get_unique_genome_id_list()
        genome_ids.sort()

        truncated_genome_data = GenomeDataCollector()

        # limit it to the first genome IDs
        for gid in genome_ids[:truncate_to]:
            info = genome_data_collector.get_info(gid)
            truncated_genome_data.add_population_member(gid)
            truncated_genome_data.set_population_member_info(gid, info)

        genome_data_collector = truncated_genome_data


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

    if reduction_type != 'nn':

        if reduction_type=='mds':
            reduction_function = reduce_using_mds
        elif reduction_type=='pca':
            reduction_function = reduce_using_pca
        else:
            raise ValueError(f"reduction_type: {reduction_type} not valid.")

        # dimensionality reduction
        reduced_genome_data = ReducedGenomeData.perform_reduction(
            source=genome_data_collector,
            dim_reduction_function=reduction_function)

    else:
        # dimensionality reduction
        reduced_genome_data = ReducedGenomeData.perform_reduction_nn(
            source=genome_data_collector,
            model_save_fname="model")

    return reduced_genome_data


def main_single(data_source, reduction_type, use_gene_data, use_weight_data, truncate_to):
    """
    Load the genome data, reduce, then save.

    Args:
        data_source: The data source.
        reduction_type: The method used for reduction.
        use_gene_data: Whether to include gene data.
        use_weight_data: Whether to include weight data.
        truncate_to: The size to truncate the data down to.
    """

    if data_source == "_":
        data_source = input("Please enter the path to retrieve the data from: ")

    data_path = os.path.join("formatted_data", data_source)

    reduced_genome_data = get_reduced(
        data_path, reduction_type, use_gene_data, use_weight_data, truncate_to)

    # set the arguments used to identify it
    identifying_args = {
        "data_source": data_source,
        "reduction_type": reduction_type,
        "use_gene_data": use_gene_data,
        "use_weight_data": use_weight_data,
        "truncate_to": truncate_to
    }

    source_name = os.path.splitext(data_source)[0]

    # make the directory
    os.makedirs("reduced_data", exist_ok=True)
    reduced_genome_data.save(
        f"reduced_data/reduced_data_{source_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.zip",
        identifying_args=identifying_args)


def main_multi(reduction_type, use_gene_data, use_weight_data, truncate_to, **kwargs):

    data_dir = "formatted_data"

    fnames = [fname for fname in os.listdir(data_dir) if os.path.splitext(fname)[1].lower() == '.zip']

    if len(fnames) == 0:
        print("No data saves available.")
        return

    for fname in fnames:
        main_single(fname, reduction_type, use_gene_data, use_weight_data, truncate_to)


if __name__=="__main__":

    multi = False

    set_config_defaults("defaults.yaml")

    with open("config.yaml", 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)

    if multi:
        main_multi(**args)
    else:
        main_single(**args)
