from genome_data import GenomeData
from local_util.load_settings import get_data_source_path, get_data_source_type, get_reduction_type, get_program_arguments
from local_util.load_examm_data import load_genomes
import os
import datetime

def get_save_fpath(reduction_type, data_source_type, add_timestamp: bool):

    save_fpath = f"vis_output/{reduction_type}-{data_source_type}"

    if add_timestamp:
        # get the timestamp for files
        dt_now = datetime.datetime.now()
        timestamp = f"{dt_now.second}-{dt_now.minute}-{dt_now.hour}-{dt_now.day}-{dt_now.month}-{dt_now.year}"
        save_fpath += "_" + timestamp

    return save_fpath


def main():

    # where to get the data_storage from
    data_source_dir = get_data_source_path()

    # what kind of dimensionality reduction to do
    reduction_type = get_reduction_type()

    # data_storage run type
    data_source_type = get_data_source_type()

    # where to store the genome data_storage
    load_fpath = f"data_storage/{reduction_type}-{data_source_type}_genome_data.zip"

    # load the source data_storage (from the EXAMM run)
    genome_data_list = load_genomes(data_dir=str(os.path.join(data_source_dir, data_source_type)))

    # create the genome data_storage class to be used for visuals
    genome_data = GenomeData()

    # load the processed data_storage from the directory
    genome_data.load_data(zip_fpath=load_fpath)

    # set colors
    genome_groups = {data_entry["generation_number"]: data_entry["group"] for data_entry in genome_data_list}
    genome_data.set_genome_colors_by_group(genome_groups)

    # extract the program arguments from settings
    args = get_program_arguments()

    save_fpath = get_save_fpath(
        reduction_type=reduction_type,
        data_source_type=data_source_type,
        add_timestamp=args["add_timestamp_to_vis"])

    # get the type of visualization to perform: 2D or 3D
    visualization_type = args["visualization_type"]

    if visualization_type == '2D':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        # do 2D visualizations
        genome_data.visualize_genomes2D(
            f"{save_fpath}.png", args=args)

    elif visualization_type == '3D':

        # make sure the visualization output directory exists so we can output to it
        os.makedirs("vis_output", exist_ok=True)

        # do 3D visualizations
        genome_data.visualize_genomes3D(
            f"{save_fpath}.gif", args=args)
    else:
        raise ValueError(f"visualization_type: \'{visualization_type}\' not recognized.")


if __name__=="__main__":
    main()
