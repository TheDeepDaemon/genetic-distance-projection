import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict


def load_results(dir: str, fname: str):
    contents = os.listdir(dir)

    data = []
    for subdir in contents:
        if os.path.isdir(os.path.join(dir, subdir)):
            fpath = os.path.join(dir, subdir, fname)
            with open(fpath, 'r') as file:
                data.append(json.loads(file.read()))

    return data


def plot_results(data: List[Dict]):
    df = pd.DataFrame(data)
    df = df[df['n_timesteps'] >= 10]

    for nt in df['n_timesteps'].unique():
        print(f"[{nt}]: {len(df[df['n_timesteps'] == nt])}")

    mean_df = df.groupby('n_timesteps').mean().reset_index()
    min_df = df.groupby('n_timesteps').min().reset_index()
    max_df = df.groupby('n_timesteps').max().reset_index()

    nn_corr_lower_error = mean_df['nn_correlation'] - min_df['nn_correlation']
    nn_corr_upper_error = max_df['nn_correlation'] - mean_df['nn_correlation']
    nn1l_corr_lower_error = mean_df['nn1l_correlation'] - min_df['nn1l_correlation']
    nn1l_corr_upper_error = max_df['nn1l_correlation'] - mean_df['nn1l_correlation']
    mds_corr_lower_error = mean_df['mds_correlation'] - min_df['mds_correlation']
    mds_corr_upper_error = max_df['mds_correlation'] - mean_df['mds_correlation']
    pca_corr_lower_error = mean_df['pca_correlation'] - min_df['pca_correlation']
    pca_corr_upper_error = max_df['pca_correlation'] - mean_df['pca_correlation']
    svd_corr_lower_error = mean_df['svd_correlation'] - min_df['svd_correlation']
    svd_corr_upper_error = max_df['svd_correlation'] - mean_df['svd_correlation']

    nn_dist_lower_error = mean_df['nn_dist'] - min_df['nn_dist']
    nn_dist_upper_error = max_df['nn_dist'] - mean_df['nn_dist']
    nn1l_dist_lower_error = mean_df['nn1l_dist'] - min_df['nn1l_dist']
    nn1l_dist_upper_error = max_df['nn1l_dist'] - mean_df['nn1l_dist']
    mds_dist_lower_error = mean_df['mds_dist'] - min_df['mds_dist']
    mds_dist_upper_error = max_df['mds_dist'] - mean_df['mds_dist']
    pca_dist_lower_error = mean_df['pca_dist'] - min_df['pca_dist']
    pca_dist_upper_error = max_df['pca_dist'] - mean_df['pca_dist']
    svd_dist_lower_error = mean_df['svd_dist'] - min_df['svd_dist']
    svd_dist_upper_error = max_df['svd_dist'] - mean_df['svd_dist']

    plt.figure(figsize=(10, 6))
    plt.errorbar(mean_df['n_timesteps'], mean_df['nn_correlation'],
                 yerr=[nn_corr_lower_error, nn_corr_upper_error], fmt='-o', capsize=5, label='NN Correlation')

    plt.errorbar(mean_df['n_timesteps'], mean_df['nn1l_correlation'],
                 yerr=[nn1l_corr_lower_error, nn1l_corr_upper_error], fmt='-o', capsize=5, label='Simple NN Correlation')

    plt.errorbar(mean_df['n_timesteps'], mean_df['mds_correlation'],
                 yerr=[mds_corr_lower_error, mds_corr_upper_error], fmt='-o', capsize=5, label='MDS Correlation')

    plt.errorbar(mean_df['n_timesteps'], mean_df['pca_correlation'],
                 yerr=[pca_corr_lower_error, pca_corr_upper_error], fmt='-o', capsize=5, label='PCA Correlation')

    plt.errorbar(mean_df['n_timesteps'], mean_df['svd_correlation'],
                 yerr=[svd_corr_lower_error, svd_corr_upper_error], fmt='-o', capsize=5, label='SVD Correlation')

    plt.xticks(df['n_timesteps'].unique())

    plt.xlabel('Number of Generations')
    plt.ylabel('Average Correlation')
    plt.title('Average Genome-Position Correlation vs. Number of Generations (Higher is Better)')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(mean_df['n_timesteps'], mean_df['nn_dist'],
                 yerr=[nn_dist_lower_error, nn_dist_upper_error], fmt='-o', capsize=5, label='NN Distance')

    plt.errorbar(mean_df['n_timesteps'], mean_df['nn1l_dist'],
                 yerr=[nn1l_dist_lower_error, nn1l_dist_upper_error], fmt='-o', capsize=5, label='Simple NN Distance')

    plt.errorbar(mean_df['n_timesteps'], mean_df['mds_dist'],
                 yerr=[mds_dist_lower_error, mds_dist_upper_error], fmt='-o', capsize=5, label='MDS Distance')

    plt.errorbar(mean_df['n_timesteps'], mean_df['pca_dist'],
                 yerr=[pca_dist_lower_error, pca_dist_upper_error], fmt='-o', capsize=5, label='PCA Distance')

    plt.errorbar(mean_df['n_timesteps'], mean_df['svd_dist'],
                 yerr=[svd_dist_lower_error, svd_dist_upper_error], fmt='-o', capsize=5, label='SVD Distance')

    plt.xticks(df['n_timesteps'].unique())

    plt.xlabel('Number of Generations')
    plt.ylabel('Average Distance')
    plt.title('Average Genome-Position Distance vs. Number of Generations (Lower is Better)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__=="__main__":

    data_list = load_results("figures", "run_log.txt")

    keys_of_interest = {
        'n_timesteps',
        'nn_correlation',
        'nn_dist',
        'nn1l_correlation',
        'nn1l_dist',
        'mds_correlation',
        'mds_dist',
        'pca_correlation',
        'pca_dist',
        'svd_correlation',
        'svd_dist'}

    filtered_data = [{k: d[k] for k in keys_of_interest if k in d} for d in data_list]

    plot_results(filtered_data)
