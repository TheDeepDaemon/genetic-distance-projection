import torch
from torch.utils.data import Dataset
from typing import List


class EmbeddingBagDataset(Dataset):

    def __init__(self, indices: List[List[int]], weights: List[List[float]]):
        self.data = [(idx, weight) for idx, weight in zip(indices, weights)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _single_diff(ind1, w1, ind2, w2):
    d1 = dict(zip(ind1, w1))
    d2 = dict(zip(ind2, w2))

    keys = set(d1.keys()) | set(d2.keys())

    total = 0.0
    for k in keys:
        v1 = d1.get(k, 0.0)
        v2 = d2.get(k, 0.0)
        total += (v1 - v2) ** 2

    return total**(1/2)


def _diff(first_batch, second_batch):

    diffs = [
        _single_diff(i1, w1, i2, w2)
        for (i1, w1), (i2, w2) in zip(first_batch, second_batch)
    ]
    return diffs


def _collate_bags(batch):

    flat_ind = []
    flat_w = []
    offsets = []

    cur_offset = 0
    for ind_list, weights in batch:
        offsets.append(cur_offset)
        flat_ind += ind_list
        flat_w += weights
        cur_offset += len(ind_list)

    input_indices = torch.tensor(flat_ind, dtype=torch.long)
    input_weights = torch.tensor(flat_w, dtype=torch.float32)
    offsets = torch.tensor(offsets, dtype=torch.long)
    return input_indices, input_weights, offsets


def collate_double_batch(batch):
    middle = len(batch) // 2
    end = middle * 2
    first_half = batch[:middle]
    second_half = batch[middle:end]

    distances = _diff(first_half, second_half)
    distances = torch.tensor(distances, dtype=torch.float32)

    batch1 = _collate_bags(first_half)
    batch2 = _collate_bags(second_half)

    return batch1, batch2, distances
