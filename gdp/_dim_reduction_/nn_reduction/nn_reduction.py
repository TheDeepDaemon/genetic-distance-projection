import torch
import numpy as np
from .model import GraphingModel
from .pair_training import fit
from ...__config__ import _set_kwargs_defaults_
import os


def reduce_using_neural_net(
        genome_data_mat: np.ndarray,
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

    # retrieve the genome size, should be the number of columns in the matrix
    genome_size = genome_data_mat.shape[1]

    model = GraphingModel(
        genome_size,
        hidden_layer1_size=kwargs["hidden_layer1_size"],
        hidden_layer2_size=kwargs["hidden_layer2_size"])

    model = model.to(device)

    # convert the genomes to a tensor
    genome_data_tensor = torch.tensor(genome_data_mat, dtype=torch.float32)

    # fit the model based on the positions, 2D positions should match genome distances
    fit(
        model=model,
        data=genome_data_tensor,
        device=device,
        batch_size=kwargs["batch_size"],
        epochs=kwargs["epochs"],
        lr=kwargs["learning_rate"],
        verbose=kwargs["verbose"])

    model_save_fpath = os.path.join(kwargs["model_save_dir"], model_save_fname)
    os.makedirs(kwargs["model_save_dir"], exist_ok=True)
    model.save(model_save_fpath)

    model.eval()
    with torch.no_grad():
        genome_data_tensor = genome_data_tensor.to(device)
        return model(genome_data_tensor).cpu().numpy()
