import numpy as np


def trace_path(positions: list, num_segments: int):
    """
    Trace a path of some group of genomes showing the chronological progression over time.

    Args:
        positions: A mapping of genome ID to position.
        num_segments: The number of segments that will be averaged out.
    """

    group_size = len(positions) // num_segments

    moving_avg_pos = []
    for i in range(num_segments):
        start = i * group_size

        segment = [positions[j] for j in range(start, start + group_size)]

        segment = np.array(segment)

        avg_pos = np.mean(segment, axis=0)

        moving_avg_pos.append(avg_pos)

    return np.array(moving_avg_pos)
