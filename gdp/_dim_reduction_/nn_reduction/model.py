import torch
import torch.nn as nn


class GraphingModel(nn.Module):
    """
    A simple neural network used to perform a mapping of N to 2,
    where N is the maximum number of genes.
    The 2D result is used for visualization_util purposes.
    """

    def __init__(self, genome_size, hidden_layer1_size, hidden_layer2_size):
        super().__init__()
        h1 = hidden_layer1_size
        h2 = hidden_layer2_size

        self.genome_size = genome_size

        self.layer1 = nn.EmbeddingBag(num_embeddings=genome_size, embedding_dim=h1, mode='sum')
        self.layer2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 2) # maps to 2D
        self.gamma = nn.Parameter(torch.tensor(1.0)) # init to 1

    def forward(self, x):
        indices, weights, offsets = x
        x = self.layer1(indices, offsets, per_sample_weights=weights)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.output_layer(x)
        return x * self.gamma

    def calc_gamma(self, dataloader, device):
        """
        Calculate the gamma value needed for this.
        """
        with torch.no_grad():
            total = 0.0
            count = 0
            for x1, x2, distances in dataloader:

                x1_indices, x1_weights, x1_offsets = x1
                x2_indices, x2_weights, x2_offsets = x2

                x1_indices, x1_weights, x1_offsets = \
                    x1_indices.to(device), x1_weights.to(device), x1_offsets.to(device)

                x2_indices, x2_weights, x2_offsets = \
                    x2_indices.to(device), x2_weights.to(device), x2_offsets.to(device)

                y1 = self((x1_indices, x1_weights, x1_offsets))
                y2 = self((x2_indices, x2_weights, x2_offsets))

                output_distances = torch.norm(y1 - y2, dim=1)
                ratio = distances / output_distances
                for element in ratio:
                    if not element.isnan():
                        total += torch.mean(element)
                        count += 1
            self.gamma = nn.Parameter(self.gamma * total / count)

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'genome_size': self.genome_size
        }, path)

    @staticmethod
    def load(path, args, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = GraphingModel(checkpoint['genome_size'], args.hidden_layer1_size, args.hidden_layer1_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
