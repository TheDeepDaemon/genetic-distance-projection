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
from gdp import set_config_defaults
from visualize_reduced_data import visualize_reduced_data
import yaml
import os


def main_single(data_source, reduction_type, use_gene_data, use_weight_data, truncate_to):
    """
    The second example of GDP being used with a neuroevolution algorithm.

    Args:
        data_source: The data source.
        reduction_type: The method used for reduction.
        use_gene_data: Whether to include gene data.
        use_weight_data: Whether to include weight data.
        truncate_to: The size it was truncated to.
    """

    visualize_reduced_data(data_source, reduction_type, use_gene_data, use_weight_data, truncate_to)


def main_multi(reduction_type, use_gene_data, use_weight_data, truncate_to, **kwargs):

    data_dir = "examm_neat_data"

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
