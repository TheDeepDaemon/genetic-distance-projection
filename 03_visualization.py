from genome_data import GenomeData
from local_util.load_settings import get_program_arguments
from local_util.load_examm_data import load_data
import os
import datetime
import argparse
import matplotlib


def get_save_fpath(reduction_type, data_source_type, add_timestamp: bool):

    save_fpath = f"vis_output/{reduction_type}-{data_source_type}"

    if add_timestamp:
        # get the timestamp for files
        dt_now = datetime.datetime.now()
        timestamp = f"{dt_now.year}-{dt_now.month}-{dt_now.day}-{dt_now.hour}-{dt_now.minute}-{dt_now.second}"
        save_fpath += "_" + timestamp

    return save_fpath


def get_best_genome(genome_data_list):
    global_best_id = None
    best_fitness = float('inf')

    for entry in genome_data_list:
        genome_id = entry["generation_number"]
        genome_fitness = entry["fitness"]

        if genome_fitness < best_fitness:
            global_best_id = genome_id
            best_fitness = genome_fitness

    return global_best_id


def main(data_source_path):

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # load the program arguments
    args = get_program_arguments()

    if data_source_path is None:
        # where to get the data_storage from
        data_source_path = args["data_source_path"]

    # what kind of dimensionality reduction to do
    reduction_type = args["reduction_type"]

    # data_storage run type
    data_source_type = args["run_type"]

    # where to store the genome data_storage
    load_fpath = f"data_storage/{reduction_type}-{data_source_type}_genome_data.zip"

    # load the source data_storage (from the EXAMM run)
    genome_data_list = list(load_data(data_filepath=f"{os.path.join(data_source_path, data_source_type)}.json").values())

    # create the genome data_storage class to be used for visuals
    genome_data = GenomeData()

    # load the processed data_storage from the directory
    genome_data.load_data(zip_fpath=load_fpath)

    if args["transform_to_01"]:
        best_genome = get_best_genome(genome_data_list)
        genome_data.transform_positions01(best_genome_id=best_genome, root_genome_id=1)

    # set colors
    genome_groups = {data_entry["generation_number"]: data_entry["group"] for data_entry in genome_data_list}
    genome_data.set_genome_colors_by_group(genome_groups)

    save_fpath = get_save_fpath(
        reduction_type=reduction_type,
        data_source_type=data_source_type,
        add_timestamp=args["add_timestamp_to_vis"])

    # get the type of visualization to perform: 2D or 3D
    visualization_type = args["visualization_type"]

    if visualization_type == '2D':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        file_ext = args["vis_image_type"]

        # do 2D visualizations
        genome_data.visualize_genomes2D(
            f"{save_fpath}.{file_ext}", args=args)

    elif visualization_type == '3D':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        # do 3D visualizations
        genome_data.visualize_genomes3D(
            f"{save_fpath}.gif", args=args)
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

    main(args.data_source_path)
