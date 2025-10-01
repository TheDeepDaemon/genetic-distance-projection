import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List
from .embedding_bag_dataset import EmbeddingBagDataset, collate_double_batch


def train_on_batch(x1, x2, distances, model, loss_fn, optimizer, device):
    x1_indices, x1_weights, x1_offsets = x1
    x2_indices, x2_weights, x2_offsets = x2

    x1_indices, x1_weights, x1_offsets = \
        x1_indices.to(device), x1_weights.to(device), x1_offsets.to(device)

    x2_indices, x2_weights, x2_offsets = \
        x2_indices.to(device), x2_weights.to(device), x2_offsets.to(device)

    y1 = model((x1_indices, x1_weights, x1_offsets))
    y2 = model((x2_indices, x2_weights, x2_offsets))

    output_distances = torch.norm(y1 - y2, dim=1)

    loss = loss_fn(output_distances, distances)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def fit(
        model: nn.Module,
        data_indices: List[List[int]],
        data_weights: List[List[float]],
        device,
        epochs: int=1000,
        batch_size: int=64,
        lr: float=0.001,
        verbose: bool=True
):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = EmbeddingBagDataset(data_indices, data_weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_double_batch)

    model.calc_gamma(dataloader, device)

    dataset_len = len(dataset)

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for batch1, batch2, distances in dataloader:
            total_loss += train_on_batch(batch1, batch2, distances, model, loss_fn, optimizer, device)

        avg_loss_primary = total_loss / dataset_len

        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss_primary:.4f}")
