from gdp import GenomeData
from local_util.load_config import load_program_arguments


def main():

    # load the program arguments
    args = load_program_arguments()

    # what kind of dimensionality reduction to do
    reduction_type = args["reduction_type"]

    # data_storage run type
    data_source_type = args["run_type"]

    # where to store the genome data_storage
    load_fpath = f"data_storage/{data_source_type}_genome_data--pre-loaded.zip"

    # where to store the genome data_storage
    save_fpath = f"data_storage/{reduction_type}-{data_source_type}_genome_data.zip"

    # create genome data_storage object
    genome_data = GenomeData()

    # load the pre-loaded data_storage
    genome_data.load_data(zip_fpath=load_fpath)

    # do reduction
    genome_data.reduce_genome(reduction_type=reduction_type, args=args)

    # save the data_storage and reduced data_storage
    genome_data.save_data(zip_fpath=save_fpath)


if __name__=="__main__":
    main()
