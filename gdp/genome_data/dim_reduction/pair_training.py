import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def fit_p(
        model: nn.Module,
        data: torch.Tensor,
        device,
        epochs: int=1000,
        batch_size: int=64,
        lr: float=0.001,
        verbose: bool=True
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
