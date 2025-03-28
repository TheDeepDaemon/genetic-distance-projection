from gdp import GenomeData, GenomeVisualizer, GenomeDataCollector
from local_util import load_program_arguments, get_subset
import os
import datetime
import argparse
import matplotlib


def get_save_fpath(reduction_type, data_source_type, add_timestamp: bool):

    save_fpath = f"vis_output/{reduction_type}-{data_source_type}"

    if add_timestamp:
        # get the timestamp for files
        dt_now = datetime.datetime.now()

        # use a timestamp down to the second to prevent overlap or conflicting names
        timestamp = f"{dt_now.year}-{dt_now.month}-{dt_now.day}-{dt_now.hour}-{dt_now.minute}-{dt_now.second}"
        save_fpath += "_" + timestamp

    return save_fpath


def get_best_genome(fitnesses):
    global_best_id = None
    best_fitness = float('inf')

    for genome_id, genome_fitness in fitnesses.items():

        if genome_fitness < best_fitness:
            global_best_id = genome_id
            best_fitness = genome_fitness

    return global_best_id


def main(data_source_path):

    # data directory name
    reduced_data_dir = "reduced_genome_data"

    # load the program arguments
    args = load_program_arguments()

    if data_source_path is None:
        # where to get the data_storage from
        data_source_path = args["data_source_path"]

    # what kind of dimensionality reduction to do
    reduction_type = args["reduction_type"]

    # data storage run type
    run_type = args["run_type"]

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
    load_fname = GenomeData.find_latest_genome_data(
        data_dir=reduced_data_dir, identifying_args=get_subset(args, identifying_keys))

    data_collector = GenomeDataCollector()
    data_collector.load(f"{os.path.join(data_source_path, run_type)}.zip")

    # create the genome data storage class to be used for visuals
    genome_data = GenomeData()

    # load the processed data storage from the directory
    genome_data.load_data(
        zip_fpath=os.path.join(reduced_data_dir, load_fname),
        identifying_args=get_subset(args, identifying_keys))

    if args["transform_to_01"]:
        fitnesses = data_collector.get_genome_attribute_by_key("fitness")
        best_genome = get_best_genome(fitnesses)
        genome_data.transform_positions01(best_genome_id=best_genome)

    # create the visualizer
    genome_visualizer = GenomeVisualizer(genome_data=genome_data)

    # set colors
    genome_groups = data_collector.get_genome_attribute_by_key("group")
    genome_visualizer.set_genome_colors_by_group(genome_groups)

    save_fpath = get_save_fpath(
        reduction_type=reduction_type,
        data_source_type=run_type,
        add_timestamp=args["add_timestamp_to_vis"])

    # get the type of visualization to perform: 2D or 3D
    visualization_type = args["visualization_type"]

    if visualization_type == '2D':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        file_ext = args["vis_image_type"]

        # do 2D visualizations
        genome_visualizer.visualize_genomes2D(save_fpath=f"{save_fpath}.{file_ext}", args=args)

    elif visualization_type == '3D':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        # do 3D visualizations
        genome_visualizer.visualize_genomes3D(f"{save_fpath}.gif", args=args)

    elif visualization_type == 'microscope':

        # do the interactive visualization
        #genome_visualizer.visualize_genomes_microscope(args=args, genome_data_dict=genome_data_dict)
        pass

    else:
        raise ValueError(f"visualization_type: \'{visualization_type}\' not recognized.")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Visualize the genome data.")

    parser.add_argument(
        "data_source_path",
        nargs="?",
        default=None,
        type=str,
        help="Filename of the directory containing EA run data."
    )

    args = parser.parse_args()

    # set the fonts for matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    main(args.data_source_path)
