import torch
import numpy as np
from .model import GraphingModel
from .pair_training import fit
from ...__config__ import _set_kwargs_defaults_
import os
from typing import List
from torch.utils.data import DataLoader
from .embedding_bag_dataset import EmbeddingBagDataset, _collate_bags


def reduce_using_neural_net(
        data_indices: List[List[int]],
        data_weights: List[List[float]],
        genome_size: int,
        model_save_fname: str,
        **kwargs):
    """
    Use a neural network to perform a mapping of N to 2 dimensions for all genomes, if N is the number of genes.

    Args:
        genome_data_mat: The genome data_storage in the form of a real-numbered matrix.
        model_save_fname: The name of the model save.

    Returns: The reduced genome data_storage.
    """
    _set_kwargs_defaults_(kwargs)

    kwargs.setdefault("batch_size", 64)
    kwargs.setdefault("epochs", 1)
    kwargs.setdefault("learning_rate", 0.0001)
    kwargs.setdefault("verbose", True)
    kwargs.setdefault("model_save_dir", "saved_models")
    kwargs.setdefault("hidden_layer1_size", 256)
    kwargs.setdefault("hidden_layer2_size", 128)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = GraphingModel(
        genome_size,
        hidden_layer1_size=kwargs["hidden_layer1_size"],
        hidden_layer2_size=kwargs["hidden_layer2_size"])

    model = model.to(device)

    # fit the model based on the positions, 2D positions should match genome distances
    fit(
        model=model,
        data_indices=data_indices,
        data_weights=data_weights,
        device=device,
        batch_size=kwargs["batch_size"],
        epochs=kwargs["epochs"],
        lr=kwargs["learning_rate"],
        verbose=kwargs["verbose"])

    model_save_fpath = os.path.join(kwargs["model_save_dir"], model_save_fname)
    os.makedirs(kwargs["model_save_dir"], exist_ok=True)
    model.save(model_save_fpath)


    dataset = EmbeddingBagDataset(data_indices, data_weights)
    dataloader = DataLoader(dataset, batch_size=len(data_indices), shuffle=True, collate_fn=_collate_bags)


    model.eval()
    with torch.no_grad():
        for input_indices, input_weights, offsets in dataloader:
            input_indices, input_weights, offsets = \
                input_indices.to(device), input_weights.to(device), offsets.to(device)
            return model((input_indices, input_weights, offsets)).cpu().numpy()
