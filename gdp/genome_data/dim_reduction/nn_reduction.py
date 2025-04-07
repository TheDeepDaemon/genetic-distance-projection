import torch
import numpy as np
from .models import GraphingModel, SimpleGraphingModel
from .pair_training import fit_p
from ...program_arguments import ProgramArguments


def reduce_using_neural_net(
        genome_data_mat: np.ndarray, args: ProgramArguments, model_type: str='standard'):
    """
    Use a neural network to perform a mapping of N to 2 dimensions for all genomes, if N is the number of genes.

    Args:
        genome_data_mat: The genome data_storage in the form of a real-numbered matrix.
        model_type: The type of model used: simple or standard.
        args: Program arguments.

    Returns: The reduced genome data_storage.
    """

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # retrieve the genome size, should be the number of columns in the matrix
    genome_size = genome_data_mat.shape[1]

    # create the model
    if model_type=='standard':
        model = GraphingModel(genome_size)
    elif model_type=='simple':
        model = SimpleGraphingModel(genome_size)
    else:
        raise ValueError(f"Model type: {model_type} not valid.")

    model = model.to(device)

    # convert the genomes to a tensor
    genome_data_tensor = torch.tensor(genome_data_mat, dtype=torch.float32)

    # fit the model based on the positions, 2D positions should match genome distances
    fit_p(
        model=model,
        data=genome_data_tensor,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        verbose=args.verbose)

    model.eval()
    with torch.no_grad():
        genome_data_tensor = genome_data_tensor.to(device)
        return model(genome_data_tensor).cpu().numpy()


def reduce_using_simple_neural_net(genome_data_mat: np.ndarray, args: ProgramArguments):
    return reduce_using_neural_net(genome_data_mat=genome_data_mat, args=args, model_type='simple')
