import numpy as np


def apply_transformation_to01(positions, rotation_mat, offset):
    positions = {gid: (pos - offset) for gid, pos in positions.items()}
    positions = {gid: np.dot(pos, rotation_mat) for gid, pos in positions.items()}

    return positions
