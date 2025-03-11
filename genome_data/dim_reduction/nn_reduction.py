from tabnanny import verbose

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class GraphingModel(nn.Module):
    """
    A simple neural network used to perform a mapping of N to 2,
    where N is the maximum number of genes.
    The 2D result is used for visualization_util purposes.
    """

    def __init__(self, genome_size):
        super(GraphingModel, self).__init__()
        h1 = 256
        h2 = 128

        self.layer1 = nn.Linear(genome_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 2)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.layer1(x))
        x = torch.nn.functional.leaky_relu(self.layer2(x))
        x = self.output_layer(x)
        return x * self.gamma


    def calc_gamma(self, dataloader, device):
        """
        Calculate the gamma value needed for this.
        """
        with torch.no_grad():
            total = 0.0
            count = 0
            for x_batch_ in dataloader:
                x_batch = x_batch_[0]
                half_ = len(x_batch) // 2
                x1 = x_batch[:half_]
                x2 = x_batch[half_:]
                if len(x1) != len(x2):
                    break

                x1 = x1.to(device)
                x2 = x2.to(device)

                y1 = self(x1)
                y2 = self(x2)

                input_distances = torch.norm(x1 - x2, dim=1)
                output_distances = torch.norm(y1 - y2, dim=1)
                ratio = input_distances / output_distances
                for element in ratio:
                    if not element.isnan():
                        total += torch.mean(element)
                        count += 1
            self.gamma = nn.Parameter(self.gamma * total / count)



class SimpleGraphingModel(nn.Module):
    """
    A simple neural network used to perform a mapping of N to 2,
    where N is the maximum number of genes.
    The 2D result is used for visualization_util purposes.
    """

    def __init__(self, genome_size):
        super(SimpleGraphingModel, self).__init__()

        self.linear_layer = nn.Linear(genome_size, 2)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.linear_layer(x)
        return x * self.gamma


    def calc_gamma(self, dataloader, device):
        """
        Calculate the gamma value needed for this.
        """
        with torch.no_grad():
            total = 0.0
            count = 0
            for x_batch_ in dataloader:
                x_batch = x_batch_[0]
                half_ = len(x_batch) // 2
                x1 = x_batch[:half_]
                x2 = x_batch[half_:]
                if len(x1) != len(x2):
                    break

                x1 = x1.to(device)
                x2 = x2.to(device)

                y1 = self(x1)
                y2 = self(x2)

                input_distances = torch.norm(x1 - x2, dim=1)
                output_distances = torch.norm(y1 - y2, dim=1)
                ratio = input_distances / output_distances
                for element in ratio:
                    if not element.isnan():
                        total += torch.mean(element)
                        count += 1
            self.gamma = nn.Parameter(self.gamma * total / count)


def fit(
    model: nn.Module, data: torch.Tensor, device,
    batch_size: int=64, epochs: int=1000, lr: float=0.001, verbose: bool=False
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    data_tensor = data.clone().detach()

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.calc_gamma(dataloader, device)

    for epoch in range(epochs):
        model.train()

        total_loss_primary = 0.0
        total_batches = 0

        for x_batch_ in dataloader:
            x_batch = x_batch_[0]
            half_ = len(x_batch) // 2
            x1 = x_batch[:half_]
            x2 = x_batch[half_:]
            if len(x1) != len(x2):
                break

            x1 = x1.to(device)
            x2 = x2.to(device)

            y1 = model(x1)
            y2 = model(x2)

            input_distances = torch.norm(x1 - x2, dim=1)
            output_distances = torch.norm(y1 - y2, dim=1)

            loss = loss_fn(output_distances, input_distances)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_primary += loss.item()
            total_batches += 1

        avg_loss_primary = total_loss_primary / total_batches

        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss_primary:.4f}")


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

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # retrieve the genome size, should be the number of columns in the matrix
    genome_size = genome_data_mat.shape[1]

    model = None

    # create the model
    if model_type=='standard':
        model = GraphingModel(genome_size)
    elif model_type=='simple':
        model = SimpleGraphingModel(genome_size)

    if model is not None:
        model = model.to(device)

        # convert the genomes to a tensor
        genome_data_tensor = torch.tensor(genome_data_mat, dtype=torch.float32)

        # fit the model based on the positions, 2D positions should match genome distances
        fit(model, genome_data_tensor, device=device, batch_size=16, epochs=epochs, lr=0.0001, verbose=verbose)

        model.eval()
        with torch.no_grad():
            genome_data_tensor = genome_data_tensor.to(device)
            return model(genome_data_tensor).cpu().numpy()

def reduce_using_simple_neural_net(genome_data_mat: np.ndarray, args):
    return reduce_using_neural_net(genome_data_mat=genome_data_mat, args=args, model_type='simple')
