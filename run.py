import numpy as np
import os
from datetime import datetime
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from visualization import create_and_save_figure, save_log
from gene_data_processing import (
    convert_genes_to_numerical, combine_gene_mats, to_color, dict_values_to_percentiles,
    reduce_genome, get_all_distances, compare_distances)
from load_data import load_data_path, load_genomes, unpack_data, get_tree_depth
from constants import VISUAL_NODE_SIZE


def convert_to_numerical(node_genes, edge_genes):
    # take the list of gene IDs and convert to a (float) vector format
    node_genes_matrix, node_genome_id_to_index = convert_genes_to_numerical(node_genes)

    # take the list of gene IDs and convert to a (float) vector format
    edge_genes_matrix, edge_genome_id_to_index = convert_genes_to_numerical(edge_genes)

    # combine the genes matrix, so that both node and edge genes are encoded
    genes_mat, genome_id_to_index = combine_gene_mats(
        genes_mat1=node_genes_matrix,
        genome_id_to_index1=node_genome_id_to_index,
        genes_mat2=edge_genes_matrix,
        genome_id_to_index2=edge_genome_id_to_index)

    return genes_mat, genome_id_to_index


def perform_visualizations(genome_data_list):
    """
    For performing visualizations at the end of a run.
    """

    # load the genomes from a temporary file
    graph, node_genes, edge_genes, fitnesses = \
        unpack_data(genome_data_list=genome_data_list)

    time_step = get_tree_depth(graph)

    # ________ ________ ________ ________ ________ ________ ________ ________ ________
    # process data

    genes_mat, genome_id_to_index = convert_to_numerical(
        node_genes=node_genes, edge_genes=edge_genes)

    # ________ ________ ________ ________ ________ ________ ________ ________ ________

    # find the best fitness, so we can mark the best genome
    best_fitness = float('inf')
    best_genome_id = -1
    for genome_id, fitness in fitnesses.items():
        if fitness < best_fitness:
            best_fitness = fitness
            best_genome_id = genome_id

    # ________ ________ ________ ________ ________ ________ ________ ________ ________

    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    # set the subdirectory name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cur_run_directory = os.path.join(figures_dir, f"run_results_{current_time}")
    if not os.path.exists(cur_run_directory):
        os.mkdir(cur_run_directory)

    # ________ ________ ________ ________ ________ ________ ________ ________ ________

    # get the positions by mapping to 2D
    nn_positions = reduce_genome(
        genes_matrix=genes_mat,
        genome_id_to_index=genome_id_to_index,
        reduction_method='nn',
        reduced_size=2,
        reduction_type='position',
        best_genome_id=best_genome_id)


    gen_distances, nn_distances = get_all_distances(
        genes_mat=genes_mat, positions=nn_positions, genome_id_to_index=genome_id_to_index)

    nn_corr = np.corrcoef(gen_distances, nn_distances)[0, 1]
    print(f"nn correlation: {nn_corr}")
    nn_comp_dist = compare_distances(gen_distances, nn_distances)
    print(f"scaled distances: {nn_comp_dist}")
    print()


    make_figures(
        graph=graph,
        positions=nn_positions,
        genes_mat=genes_mat,
        genome_id_to_index=genome_id_to_index,
        fitnesses=fitnesses,
        time_step=time_step,
        cur_run_directory=cur_run_directory,
        type='nn')

    # ________ ________ ________ ________ ________ ________ ________ ________ ________

    # get the positions by mapping to 2D
    nn1l_positions = reduce_genome(
        genes_matrix=genes_mat,
        genome_id_to_index=genome_id_to_index,
        reduction_method='nn1l',
        reduced_size=2,
        reduction_type='position',
        best_genome_id=best_genome_id)

    gen_distances, nn1l_distances = get_all_distances(
        genes_mat=genes_mat, positions=nn1l_positions, genome_id_to_index=genome_id_to_index)

    nn1l_corr = np.corrcoef(gen_distances, nn1l_distances)[0, 1]
    print(f"nn1l correlation: {nn1l_corr}")
    nn1l_comp_dist = compare_distances(gen_distances, nn1l_distances)
    print(f"scaled distances: {nn1l_comp_dist}")
    print()

    make_figures(
        graph=graph,
        positions=nn1l_positions,
        genes_mat=genes_mat,
        genome_id_to_index=genome_id_to_index,
        fitnesses=fitnesses,
        time_step=time_step,
        cur_run_directory=cur_run_directory,
        type='nn1l')

    # ________ ________ ________ ________ ________ ________ ________ ________ ________

    mds_positions = reduce_genome(
        genes_matrix=genes_mat,
        genome_id_to_index=genome_id_to_index,
        reduction_method='mds',
        reduced_size=2,
        reduction_type='position',
        best_genome_id=best_genome_id)

    gen_distances, mds_distances = get_all_distances(
        genes_mat=genes_mat, positions=mds_positions, genome_id_to_index=genome_id_to_index)

    mds_corr = np.corrcoef(gen_distances, mds_distances)[0, 1]
    print(f"mds correlation: {mds_corr}")
    mds_comp_dist = compare_distances(gen_distances, mds_distances)
    print(f"scaled distances: {mds_comp_dist}")
    print()

    make_figures(
        graph=graph,
        positions=mds_positions,
        genes_mat=genes_mat,
        genome_id_to_index=genome_id_to_index,
        fitnesses=fitnesses,
        time_step=time_step,
        cur_run_directory=cur_run_directory,
        type='mds')

    # ________ ________ ________ ________ ________ ________ ________ ________ ________

    pca_positions = reduce_genome(
        genes_matrix=genes_mat,
        genome_id_to_index=genome_id_to_index,
        reduction_method='pca',
        reduced_size=2,
        reduction_type='position',
        best_genome_id=best_genome_id)

    gen_distances, pca_distances = get_all_distances(
        genes_mat=genes_mat, positions=pca_positions, genome_id_to_index=genome_id_to_index)

    pca_corr = np.corrcoef(gen_distances, pca_distances)[0, 1]
    print(f"pca correlation: {pca_corr}")
    pca_comp_dist = compare_distances(gen_distances, pca_distances)
    print(f"scaled distances: {pca_comp_dist}")
    print()

    make_figures(
        graph=graph,
        positions=pca_positions,
        genes_mat=genes_mat,
        genome_id_to_index=genome_id_to_index,
        fitnesses=fitnesses,
        time_step=time_step,
        cur_run_directory=cur_run_directory,
        type='pca')

    # ________ ________ ________ ________ ________ ________ ________ ________ ________

    svd_positions = reduce_genome(
        genes_matrix=genes_mat,
        genome_id_to_index=genome_id_to_index,
        reduction_method='svd',
        reduced_size=2,
        reduction_type='position',
        best_genome_id=best_genome_id)

    gen_distances, svd_distances = get_all_distances(
        genes_mat=genes_mat, positions=svd_positions, genome_id_to_index=genome_id_to_index)

    svd_corr = np.corrcoef(gen_distances, svd_distances)[0, 1]
    print(f"svd correlation: {svd_corr}")
    svd_comp_dist = compare_distances(gen_distances, svd_distances)
    print(f"scaled distances: {svd_comp_dist}")
    print()

    make_figures(
        graph=graph,
        positions=svd_positions,
        genes_mat=genes_mat,
        genome_id_to_index=genome_id_to_index,
        fitnesses=fitnesses,
        time_step=time_step,
        cur_run_directory=cur_run_directory,
        type='svd')

    # ________ ________ ________ ________ ________ ________ ________ ________ ________

    log_data = dict()
    log_data["n_timesteps"] = time_step - 1
    log_data["total_population_size"] = graph.number_of_nodes()
    log_data["n_population_connections"] = graph.number_of_edges()
    log_data["nn_correlation"] = nn_corr
    log_data["nn_dist"] = nn_comp_dist
    log_data["nn1l_correlation"] = nn1l_corr
    log_data["nn1l_dist"] = nn1l_comp_dist
    log_data["mds_correlation"] = mds_corr
    log_data["mds_dist"] = mds_comp_dist
    log_data["pca_correlation"] = pca_corr
    log_data["pca_dist"] = pca_comp_dist
    log_data["svd_correlation"] = svd_corr
    log_data["svd_dist"] = svd_comp_dist
    save_log(log_data, fpath=os.path.join(cur_run_directory, "run_log"))


def make_figures(
        graph,
        positions,
        genes_mat,
        genome_id_to_index,
        fitnesses,
        time_step,
        cur_run_directory,
        type
):
    subdir = os.path.join(cur_run_directory, type)
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    pos_arr = np.array(list(positions.values()), dtype=float)
    pairwise_dist = squareform(pdist(pos_arr))

    # the size of the nodes in the visualization scale with the distances between them
    node_size = VISUAL_NODE_SIZE * np.mean(pairwise_dist)

    # use PCA to determine colors
    pca_colors = reduce_genome(
        genes_matrix=genes_mat,
        genome_id_to_index=genome_id_to_index,
        reduction_method='pca',
        reduced_size=3,
        reduction_type='color')

    # use timings to determine colors
    #timing_colors = {gid: (t / time_step) for gid, t in creation_times.items()}
    #timing_colors = {gid: (b, b, b) for gid, b in timing_colors.items()}

    # use fitness to determine colors
    red = (1.0, 0.0, 0.0)
    blue = (0.0, 0.0, 1.0)
    col_low = red
    col_high = blue
    fitness_colors = dict_values_to_percentiles(fitnesses=fitnesses)
    fitness_colors = {gid: to_color(f, col_low=col_low, col_high=col_high) for gid, f in fitness_colors.items()}

    # create the figures...

    legend_handles = [
        Patch(color=col_low, label='Low Loss'),
        Patch(color=col_high, label='High Loss'),
    ]

    # perform the visualizations and save the resulting figures
    create_and_save_figure(
        graph=graph,
        positions=positions,
        node_size=node_size,
        node_colors=pca_colors,
        fpath=os.path.join(subdir, "pca_colors"))

    """
    create_and_save_figure(
        graph=graph,
        positions=positions,
        node_size=node_size,
        node_colors=timing_colors,
        fpath=os.path.join(subdir, "timing_colors"))"""

    x_limits, y_limits = create_and_save_figure(
        graph=graph,
        positions=positions,
        node_size=node_size,
        node_colors=fitness_colors,
        fpath=os.path.join(subdir, "fitness_colors"),
        legend_handles=legend_handles)

def visualize_data():
    data_dir = load_data_path()
    genomes = load_genomes(data_dir=data_dir)
    perform_visualizations(genomes)

if __name__=="__main__":
    visualize_data()
