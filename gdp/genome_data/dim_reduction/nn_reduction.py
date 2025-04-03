import torch
import numpy as np
from .models import GraphingModel, SimpleGraphingModel
from .pair_training import fit_p


def parse_program_arguments(args, keyword, expected_type, default_val):
    """
    Parse a specific program argument by keyword.

    Args:
        args: The program arguments list that is passed in.
        keyword: The key to retrieve the value.
        expected_type: The type it is expected to be.
        default_val: The default value of this argument.

    Returns: The resulting argument value.
    """
    if keyword in args:
        arg_val = args[keyword]
        if isinstance(arg_val, expected_type):
            return arg_val
        elif isinstance(arg_val, str):
            return expected_type(arg_val)

    return default_val


def reduce_using_neural_net(genome_data_mat: np.ndarray, args, model_type: str='standard'):
    """
    Use a neural network to perform a mapping of N to 2 dimensions for all genomes, if N is the number of genes.

    Args:
        genome_data_mat: The genome data_storage in the form of a real-numbered matrix.
        model_type: The type of model used: simple or standard.
        args: Program arguments.

    Returns: The reduced genome data_storage.
    """

    # parsing the verbose argument
    verbose = parse_program_arguments(args, keyword="verbose", expected_type=bool, default_val=True)

    # parsing the epochs argument
    epochs = parse_program_arguments(args, keyword="epochs", expected_type=int, default_val=1)

    # parsing the batch_size argument
    batch_size = parse_program_arguments(args, keyword="batch_size", expected_type=int, default_val=64)

    # parsing the learning_rate argument
    learning_rate = parse_program_arguments(args, keyword="learning_rate", expected_type=float, default_val=0.0001)

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
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        verbose=verbose)

    model.eval()
    with torch.no_grad():
        genome_data_tensor = genome_data_tensor.to(device)
        return model(genome_data_tensor).cpu().numpy()

def reduce_using_simple_neural_net(genome_data_mat: np.ndarray, args):
    return reduce_using_neural_net(genome_data_mat=genome_data_mat, args=args, model_type='simple')
