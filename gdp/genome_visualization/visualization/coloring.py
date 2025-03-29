import numpy as np
from scipy.stats import rankdata
import math
import matplotlib.colors as mcolors
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def dict_values_to_percentiles(fitnesses: dict):
    """
    Convert the fitnesses to a ranking between 0 and 1, preserving inequalities.

    Args:
        fitnesses: The dict of genome ID to fitness.
    """

    fitnesses_real = {
        gid: fitness
        for gid, fitness in fitnesses.items()
        if not (math.isnan(fitness) or math.isinf(fitness))}

    id_to_index = {gid: idx for idx, gid in enumerate(fitnesses_real)}

    fitnesses_arr = np.zeros(len(id_to_index), dtype=float)

    for gid, idx in id_to_index.items():
        fitnesses_arr[idx] = fitnesses[gid]

    ranks = rankdata(fitnesses_arr, method='average')

    normalized_ranks = (ranks - 1) / (len(fitnesses_arr) - 1)

    percentile_dict = {gid: np.float64(1) for gid in fitnesses}
    for gid, idx in id_to_index.items():
        percentile_dict[gid] = normalized_ranks[idx]

    return percentile_dict


def to_color(
        val: float,
        col_low: Tuple[float|int, float|int, float|int],
        col_high: Tuple[float|int, float|int, float|int]
) -> Tuple[float, float, float]:
    """
    Convert a scalar to an RGB color.

    Args:
        val: The single value.
        col_low: The color for low weights
        col_high: The color for high weights.

    Returns: A tuple of RGB values.
    """
    if math.isnan(val):
        return (0.0, 0.0, 0.0)
    else:
        r = ((1.0 - val) * col_low[0]) + (val * col_high[0])
        g = ((1.0 - val) * col_low[1]) + (val * col_high[1])
        b = ((1.0 - val) * col_low[2]) + (val * col_high[2])
        return (r, g, b)


def calc_colors_by_fitness(fitness_values: dict, col_low, col_high):
    """
    Set the colors of the genomes based on fitness values.

    Args:
        fitness_values: A dict mapping genome ID to fitness.
        col_low: The color indicating low fitness.
        col_high: The color indicating high fitness.

    Returns:
        Tuple containing the dictionary of genome ID to color, and the legend_handles.
    """

    fitness_colors = dict_values_to_percentiles(fitnesses=fitness_values)
    node_colors = {gid: to_color(f, col_low=col_low, col_high=col_high) for gid, f in fitness_colors.items()}

    legend_handles = [
        Patch(color=col_low, label='Low Loss'),
        Patch(color=col_high, label='High Loss'),
    ]

    return node_colors, legend_handles


def generate_arbitrary_colors(n: int):
    """
    Generate a palette, with 'n' random colors.

    Args:
        n: The size of the color palette.

    Returns:
        A list of colors.
    """
    cmap = plt.get_cmap('hsv')
    nsq = min(n**2, 2**24)
    colors = [cmap(i / nsq) for i in range(nsq)]
    np.random.shuffle(colors)
    return colors[:n]


def calc_colors_by_group(genome_groups: dict):
    """
    Set the colors of the genomes based on what group they belong to.

    Args:
        genome_groups: The group number for each genome.

    Returns:
        Tuple containing the dictionary of genome ID to color, and the legend_handles.
    """

    groups = set(genome_groups.values())

    tableau_colors = list(mcolors.TABLEAU_COLORS.values())
    css4_colors = list(mcolors.CSS4_COLORS.values())

    n_groups = len(groups)

    # pick the color palette
    if n_groups <= len(tableau_colors):
        # use tableau colors if there aren't that many groups
        color_list = tableau_colors

    elif n_groups <= len(css4_colors):
        # use css4 colors if there are more
        color_list = css4_colors
    else:
        # use (basically) randomly generated colors if there are even more than that
        color_list = generate_arbitrary_colors(n_groups)

    shuffled_indices = np.arange(len(groups))
    np.random.shuffle(shuffled_indices)

    group_indices = {group_num: idx for idx, group_num in zip(shuffled_indices, groups)}

    colors_dict = {
        group_number: color_list[group_indices[group_number]]
        for group_number in group_indices}

    genome_colors = {
        gid: colors_dict[group_number]
        for gid, group_number in genome_groups.items()}

    legend_handles = [
        Patch(color=colors_dict[group_number], label=f"Group {group_number}")
        for group_number in list(groups)]

    return genome_colors, legend_handles
