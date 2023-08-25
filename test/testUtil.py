import torch
import torch.nn as nn

from models.transformers.layers import EncoderLayer  # isort: skip

# Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_DATASET_EPOCH = 20


def train(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epoch: int,
    optimizer: torch.optim,
    loss_fn: nn,
) -> tuple:
    """
    Train function for converage test only. No validation process
    Parameters
    ----------
    model: The model
    x_test: Input data
    y_test: Output data
    epoch: Number of epochs
    optimizer: Optimizer
    loss_fn: Loss function
    Returns
    -------
    Tuple containing model and final loss
    """
    model.train()
    model.to(device)
    for _ in range(epoch):
        y_pred = model.forward(x_test)
        loss = loss_fn(y_pred, y_test)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(model.forward(x_test), y_test)
    return model, final_loss


def train_attention(
    model: nn.Module,
    x_test1: torch.Tensor,
    x_test2: torch.Tensor,
    x_test3: torch.Tensor,
    y_test: torch.Tensor,
    epoch: int,
    optimizer: torch.optim,
    loss_fn: nn,
) -> tuple:
    """
    Train function for converage test for attention only. No validation process
    Parameters
    ----------
    model: The model
    x_test1 -> x_test3: Input data
    y_test: Output data
    epoch: Number of epochs
    optimizer: Optimizer
    loss_fn: Loss function
    Returns
    -------
    Tuple containing model and final loss
    """
    model.train()
    model.to(device)
    for _ in range(epoch):
        y_pred = model.forward(x_test1, x_test2, x_test3)
        loss = loss_fn(y_pred, y_test)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(model.forward(x_test1, x_test2, x_test3), y_test)
    return model, final_loss


if __name__ == "__main__":
    model = EncoderLayer(10, 5, 16)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _, final_loss = train(
        model, torch.rand(5, 20, 10), torch.rand(5, 20, 10), 10, optimizer, loss_fn
    )
    print(f"Final loss: {final_loss}")
