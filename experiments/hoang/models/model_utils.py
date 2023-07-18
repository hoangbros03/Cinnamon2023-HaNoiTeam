import logging

import torch.nn as nn

# Logger
logging.basicConfig(format="%(asctime)s %(message)s")
log = logging.getLogger("test_logger")
log.setLevel(logging.DEBUG)


def define_activation(typeActivation: str = "sigmoid") -> None:
    """
    Define the activation function for this class
    Parameters
    ----------
    typeActivation: type of activation function
    Returns
    -------
    Nothing
    """
    activation1: str = typeActivation
    if activation1 == "sigmoid":
        return nn.Sigmoid()
    elif activation1 == "tanh":
        return nn.Tanh()
    elif activation1 == "relu":
        return nn.ReLU()
    else:
        log.error("Wrong type of activation. Change it to sigmoid...")
        return nn.Sigmoid()


def init_params(weights: list, type_init: str = "xavier_uniform") -> list:
    """
    Init the parameters of model
    Parameters
    ----------
    weights: List of weight need to be initialized
    type: How the weights be initialized
    """
    if type(weights) != list or type(type_init) != str:
        raise TypeError("Wrong type(s) of parameter(s)")
    init = None
    if type_init == "xavier_uniform":
        init = nn.init.xavier_uniform_
    elif type_init == "xavier_normal":
        init = nn.init.xavier_normal_
    elif type_init == "kaiming_uniform":
        init = nn.init.kaiming_uniform_
    elif type_init == "kaiming_normal":
        init = nn.init.kaiming_normal_
    else:
        raise ValueError("Type not supported.")
    for w in range(len(weights)):
        weights[w] = init(weights[w])
    return weights
