import torch
import torch.nn as nn


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
