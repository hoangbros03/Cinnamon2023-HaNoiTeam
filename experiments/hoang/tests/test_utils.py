import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import experiments.hoang.models.rnn_htb as rnn_htb

# Logging config
logging.basicConfig(format="%(asctime)s %(message)s")
log = logging.getLogger("test_logger")
log.setLevel(logging.DEBUG)

# Constant variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Necessary function
def train(
    model: rnn_htb.RNN,
    X_train: torch.tensor,
    y_train: torch.tensor,
    loss_fn: nn,
    optimizer: torch.optim,
    verbose: bool = False,
) -> rnn_htb.RNN:
    """
    Train function
    Parameters
    ----------
    model: Model instance
    X_train: X train dataset
    y_train: y train dataset
    loss_fn: Loss function
    optimizer: Optimizer
    verbose: Choose to see log or not
    Returns
    -------
    Model trained
    """
    # Switch to train mode
    model.train()

    # Transfer to device
    model.to(device)
    X_train.to(device)
    y_train.to(device)

    # Train step
    optimizer.zero_grad()
    y_pred = model.forward(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # Log the loss and acc
    if verbose:
        verbose_result(y_pred, y_train, loss=loss)
    return model


def eval(
    model: rnn_htb.RNN,
    X_test: torch.tensor,
    y_test: torch.tensor,
    loss_fn: nn = nn.CrossEntropyLoss,
    verbose: bool = True,
) -> None:
    """
    Evaluation function to evaluate model
    Parameters
    ----------
    model: A RNN model
    X_test: X test dataset
    y_test: y test dataset
    loss_fn: Loss function
    verbose: Choose to see log or not
    Returns
    -------
    Nothing
    """
    # Switch to train mode
    model.eval()

    # Transfer to device
    model.to(device)
    X_test.to(device)
    y_test.to(device)

    # Test step
    with torch.inference_mode:
        y_pred = model.forward(X_test)
        loss = loss_fn(y_pred, y_test)
        if verbose:
            verbose_result(y_pred, y_test, loss=loss)


def verbose_result(
    y_pred, y_test, loss=None, many: bool = False, loss_fn: nn = nn.CrossEntropyLoss
) -> None:
    """
    Function to get log of loss and accuracy
    This is made to re-use in many necessary steps
    Parameters
    ----------
    y_pred: y prediction from model
    y_test: y label from dataset
    loss: Use if loss is calculated. It's none by default
    many: Use if RNN model output is "many", not "one"
    loss_fn: Loss function. It's skipped if loss is provided
    Returns
    -------
    Nothing
    """
    if loss is None:
        loss = loss_fn(y_pred, y_test)
    if not many:
        loss_log = loss / len(y_test)
        # log.debug(f"DEBUG 1: {torch.eq(torch.argmax(y_pred,dim=1), y_train)}")
        # log.debug(f"DEBUG 2: {torch.eq(torch.argmax(y_pred,dim=1), y_train).sum()}")
        # log.debug(
        # f"DEBUG 3: {torch.eq(torch.argmax(y_pred,dim=1), y_train).sum().item()}"
        # )
        acc_log = (
            torch.eq(torch.argmax(y_pred, dim=1), y_test).sum().item()
            * 1.0
            / len(y_test)
        )
        log.info(f"Loss: {loss_log}, acc: {acc_log}")
    else:
        raise Exception("Developing function...")


def get_data(name: str) -> tuple:
    """
    Get data from scikit learn library
    Parameters
    ----------
    name: "short" name of dataset
    Returns
    -------
    np.float32 dataset, which is compatible with our RNN model
    """
    if name == "iris":
        iris = load_iris()
        iris_x = pd.DataFrame(iris.data)
        iris_y = pd.DataFrame(iris.target)
        return np.float32(iris_x.to_numpy()), np.float32(iris_y.to_numpy())
    else:
        raise Exception("name passed is not supported!")


def split_dataset(
    X: np.array, y: np.array, test_size: float = 0.3, val_size: float = 0.3
) -> tuple:
    """
    Split the dataset into train, val, test
    Parameters
    ----------
    X: input data
    y: label data
    test_size: Size of test dataset
    val_size: Size of validation dataset. Val_size can be None!
    Returns
    -------
    A tuple containing train, val, test with X,y dataset respectively
    """
    # Check if test size greater than 0
    if test_size <= 0:
        raise Exception("Test size can't below or equal to 0")

    # Split
    X_val, y_val = None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Check val size is none
    if val_size is None:
        return X_train, y_train, X_val, y_val, X_test, y_test

    # Check if val size <= 0
    if val_size <= 0:
        raise Exception(
            "Validation size can't below or equal to 0. Please set it to None if you"
            " don't want validation dataset."
        )

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_model(
    name: str,
    input_size: int,
    hidden_size: int,
    output_size: int,
    input_times: int = 0,
    output_times: int = 0,
    bias: bool = False,
    activation1: str = "sigmoid",
    activation2: str = "sigmoid",
    simultaneous: bool = False,
) -> object:
    """
    Get model with specfied name and parameters
    Use it as middleman between real instance
    and test class.
    Note that you don't have to set redundant parameters
    if your selected model doesn't use them.
    Parameters
    ----------
    name: Name of the model
    input_size: Input dims of input data
    hidden_size: Hidden size of a<t>
    output_size: Output dims of output data
    input_times: Times of input
    output_times: Times of output
    bias: Choose if you want bias or not
    activation1: activation function to get a<t>
    activation2: activation function to get y<t>
    simultaneous: Use for many to many RNN only
    Returns
    -------
    Instance of selected model
    """
    if name == "rnn" or name == "RNN":
        return rnn_htb.RNN(
            input_size, hidden_size, output_size, bias, activation1, activation2
        )
    elif name == "oneToManyRNN":
        return rnn_htb.oneToManyRNN(
            output_times,
            input_size,
            hidden_size,
            output_size,
            bias,
            activation1,
            activation2,
        )
    elif name == "manyToOneRNN":
        return rnn_htb.manyToOneRNN(
            input_times,
            input_size,
            hidden_size,
            output_size,
            bias,
            activation1,
            activation2,
        )
    elif name == "manyToManyRNN":
        return rnn_htb.manyToManyRNN(
            input_times,
            output_times,
            input_size,
            hidden_size,
            output_size,
            bias,
            activation1,
            activation2,
            simultaneous,
        )
    else:
        raise Exception("unsupported type of RNN!")


def get_and_process_data(
    name: str,
    test_size: float = 0.3,
    val_size: float = 0.3,
) -> tuple:
    """
    Function that combined the data process steps
    Note that label will be change to long type so it can be handled by
    nn.CrossEntropyLoss()
    Parameters
    ----------
    name: name of the dataset
    test_size: size of test dataset
    val_szie: size of validation dataset
    Returns
    -------
    Tuple of X, y dataset
    """
    # Get data
    X, y = get_data(name)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y, test_size, val_size
    )

    if name == "iris":
        # Process the data
        X_train, y_train, X_test, y_test = (
            torch.tensor(X_train),
            torch.tensor(y_train).squeeze().long(),
            torch.tensor(X_test),
            torch.tensor(y_test).squeeze().long(),
        )

        if X_val is not None and y_val is not None:
            X_val, y_val = (torch.tensor(X_val), torch.tensor(y_val).squeeze().long())
        log.debug(f"X_train shape: {X_train.shape}")
        log.debug(f"y_train shape: {y_train.shape}")
    else:
        raise Exception("unsupported dataset.")

    return X_train, y_train, X_val, y_val, X_test, y_test
