import numpy as np
from ..gdp import GenomeDataCollector
from ..program_arguments import ProgramArguments
import matplotlib.pyplot as plt
import matplotlib

_run_type_titles = {
    "evoviz_examples_neat_high_speciation_1epoch": "NEAT High Speciation 1 BP Epoch",
    "evoviz_examples_neat_high_speciation_10epochs": "NEAT High Speciation 10 BP Epochs",
    "evoviz_examples_neat_low_speciation_1epoch": "NEAT Low Speciation 1 BP Epoch",
    "evoviz_examples_neat_low_speciation_10epochs": "NEAT Low Speciation 10 BP Epochs",
    "evoviz_examples_no_repop_1epoch": "EXAMM No Repopulation 1 BP Epoch",
    "evoviz_examples_no_repop_10epochs": "EXAMM No Repopulation 10 BP Epochs",
    "evoviz_examples_repop_1epoch": "EXAMM With Repopulation 1 BP Epoch",
    "evoviz_examples_repop_10epochs": "EXAMM With Repopulation 10 BP Epochs",
}


def plot_genes_vs_generation_number():

    # set the fonts for matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # load the program arguments
    program_args = ProgramArguments()

    data_collector = GenomeDataCollector()
    data_collector.load(f"data/{program_args.run_type}.zip")

    genome_ids = data_collector.get_unique_genome_id_list()
    genome_ids.sort()

    all_node_ids = data_collector.get_genome_attribute_by_key("nodes", "n")
    all_edge_ids = data_collector.get_genome_attribute_by_key("edges", "n")
    add_recurrent_edge_ids = data_collector.get_genome_attribute_by_key("recurrent_edges", "n")

    node_ids = set()
    edge_ids = set()
    redge_ids = set()

    n_tracker = []
    e_tracker = []
    re_tracker = []

    for gid in genome_ids:
        for n_id in all_node_ids[gid]:
            node_ids.add(n_id)

        for e_id in all_edge_ids[gid]:
            edge_ids.add(e_id)

        for re_id in add_recurrent_edge_ids[gid]:
            redge_ids.add(re_id)

        n_tracker.append(len(node_ids))
        e_tracker.append(len(edge_ids))
        re_tracker.append(len(redge_ids))

    genome_ids = np.array(genome_ids)
    n_tracker = np.array(n_tracker)
    e_tracker = np.array(e_tracker)
    re_tracker = np.array(re_tracker)

    plt.plot(genome_ids, n_tracker, label='# unique nodes')
    plt.plot(genome_ids, e_tracker, label='# unique edges')
    plt.plot(genome_ids, re_tracker, label='# unique recurrent edges')
    plt.plot(genome_ids, np.add(n_tracker, e_tracker, re_tracker), label='# total unique genes')

    plt.xlabel('Genomes Created')
    plt.ylabel('Number of Gene IDs')
    plt.title(_run_type_titles[program_args.run_type])

    plt.legend()

    plt.savefig(f"unique_genes_{program_args.run_type}.pdf")
