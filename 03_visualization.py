import torch

from gdp import GenomeMatrix, GenomeVisualizer, GenomeDataCollector
from gdp.program_arguments import ProgramArguments
import os
import datetime
import matplotlib
import json


run_type_titles = {
    "evoviz_examples_neat_high_speciation_1epoch": "NEAT High Speciation 1 BP Epoch",
    "evoviz_examples_neat_high_speciation_10epochs": "NEAT High Speciation 10 BP Epochs",
    "evoviz_examples_neat_low_speciation_1epoch": "NEAT Low Speciation 1 BP Epoch",
    "evoviz_examples_neat_low_speciation_10epochs": "NEAT Low Speciation 10 BP Epochs",
    "evoviz_examples_no_repop_1epoch": "EXAMM No Repopulation 1 BP Epoch",
    "evoviz_examples_no_repop_10epochs": "EXAMM No Repopulation 10 BP Epochs",
    "evoviz_examples_repop_1epoch": "EXAMM With Repopulation 1 BP Epoch",
    "evoviz_examples_repop_10epochs": "EXAMM With Repopulation 10 BP Epochs",
}


def save_config(save_fpath: str, args: dict):
    """
    Save information about the config used for these visuals.
    Args:
        save_fpath: The filename used for the visuals.
        args: The program arguments.
    """
    _, fname = os.path.split(save_fpath)
    fname_no_ext, f_ext = os.path.splitext(fname)
    with open(f"config_log/{fname_no_ext}.json", 'w', encoding='utf-8') as f:
        args["output_fname"] = fname
        json.dump(args, f, indent=4)


def get_save_fpath(reduction_type, data_source_type, add_timestamp: bool):

    save_fpath = f"vis_output/{reduction_type}-{data_source_type}"

    if add_timestamp:
        # get the timestamp for files
        dt_now = datetime.datetime.now()

        # use a timestamp down to the second to prevent overlap or conflicting names
        timestamp = f"{dt_now.year}-{dt_now.month}-{dt_now.day}-{dt_now.hour}-{dt_now.minute}-{dt_now.second}"
        save_fpath += "_" + timestamp

    return save_fpath


def main():

    # load the program arguments
    program_args = ProgramArguments()

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # LOAD DATA

    identifying_keys = [
        "reduction_type",
        "run_type",
        "use_node_gene_data",
        "use_edge_gene_data",
        "use_edge_weights_data",
        "use_recurrent_edge_gene_data",
        "use_recurrent_edge_weights_data"
    ]

    # find the genome data to use
    load_fname = GenomeMatrix.find_latest_genome_data(
        data_dir="reduced_genome_data", identifying_args=program_args.get_subset(identifying_keys))

    data_collector = GenomeDataCollector()
    data_collector.load(f"data/{program_args.run_type}.zip")

    # create the genome data storage class to be used for visuals
    genome_matrix = GenomeMatrix()

    # load the processed data storage from the directory
    genome_matrix.load_data(
        zip_fpath=os.path.join("reduced_genome_data", load_fname),
        identifying_args=program_args.get_subset(identifying_keys))

    data_collector.set_reduced_positions(genome_matrix.get_positions())
    data_collector.set_unreduced_gene_vectors(genome_matrix.get_unreduced_gene_vectors())

    model = genome_matrix.get_model(args=program_args)

    #  ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
    # CREATE VISUALIZER AND DO VISUALIZATION

    # create the visualizer
    genome_visualizer = GenomeVisualizer(genome_data_collector=data_collector)

    if model is not None:
        genome_visualizer.nn_model = model

    # set colors
    genome_groups = data_collector.get_genome_attribute_by_key("group")
    genome_visualizer.set_genome_colors_by_group(genome_groups)

    save_fpath = get_save_fpath(
        reduction_type=program_args.reduction_type,
        data_source_type=program_args.run_type,
        add_timestamp=program_args.add_timestamp_to_vis)

    title = run_type_titles[program_args.run_type]

    # get the type of visualization to perform: 2D or 3D
    visualization_type = program_args.visualization_type

    if visualization_type == '2D':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        # do 2D visualizations
        save_fpath = f"{save_fpath}.{program_args.vis_image_type}"
        genome_visualizer.visualize_genomes2D(save_fpath=save_fpath, args=program_args.args)
        save_config(save_fpath=save_fpath, args=program_args.args)

    elif visualization_type == '3D':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        # do 3D visualizations
        save_fpath = f"{save_fpath}.gif"
        genome_visualizer.visualize_genomes3D_GIF(
            save_fpath=save_fpath, args=program_args, title=title)
        save_config(save_fpath=save_fpath, args=program_args.args)

    elif visualization_type == '3D-frames':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        # do 3D visualizations
        genome_visualizer.visualize_genomes3D_images(
            save_fpath=save_fpath, args=program_args, title=title)
        save_config(save_fpath=save_fpath, args=program_args.args)

    elif visualization_type == 'microscope':

        # do the interactive visualization
        genome_visualizer.visualize_genomes_microscope(args=program_args.args, data_collector=data_collector)

    else:
        raise ValueError(f"visualization_type: \'{visualization_type}\' not recognized.")


if __name__=="__main__":

    # set the fonts for matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    main()
