from genome_data import GenomeData, join_genomes
from local_util.load_settings import get_program_arguments
from local_util.load_examm_data import load_genomes
import os


def main():

    # load the program arguments
    args = get_program_arguments()

    # where to get the data_storage from
    data_source_dir = args["data_source_path"]

    # data_storage run type
    data_source_type = args["run_type"]

    # where to store the genome data_storage
    save_fpath = f"data_storage/{data_source_type}_genome_data--pre-loaded.zip"

    # load the source data_storage (from the EXAMM run)
    genome_data_list = load_genomes(data_dir=str(os.path.join(data_source_dir, data_source_type)))

    node_genes = {data_entry["generation_number"]: data_entry["nodes"] for data_entry in genome_data_list}
    edge_genes = {data_entry["generation_number"]: data_entry["edges"] for data_entry in genome_data_list}

    # genome data_storage from node genes
    node_gene_data = GenomeData()
    node_gene_data.init_data(node_genes)

    # genome data_storage from edge genes
    edge_gene_data = GenomeData()
    edge_gene_data.init_data(edge_genes)

    # get combined genome data_storage
    genome_data = join_genomes(node_gene_data, edge_gene_data)

    # set genome IDs
    genome_ids = list({data_entry["generation_number"] for data_entry in genome_data_list})

    # set parent-child relations
    relations = []
    for data_entry in genome_data_list:
        for parent in data_entry["parents"]:
            relations.append((parent, data_entry["generation_number"]))

    # init graph data_storage with genome IDs and parent-child relations
    genome_data.init_graph_data(genome_ids=genome_ids, relations=relations)

    # make sure this directory exists, so we can output to it
    os.makedirs("data_storage", exist_ok=True)

    # save the data_storage and reduced data_storage
    genome_data.save_data(zip_fpath=save_fpath)


if __name__=="__main__":
    main()
