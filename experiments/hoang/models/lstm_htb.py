# Import libs and frameworks
import logging

import torch.nn as nn

# Logger
logging.basicConfig(format="%(asctime)s %(message)s")
log = logging.getLogger("test_logger")
log.setLevel(logging.DEBUG)


class LSTM(nn.Module):
    """
    The (a) Long-short term memory class.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        activation1: str = "sigmoid",
        activation2: str = "tanh",
    ) -> None:
        """
        Constructor of the class.
        Parameters
        ----------
        input_size: embedding size of the input
        output_size: embedding size of the output
        bias: set to "True" to enable bias
        activation1: set the activation to calc the a<t>
        activation2: set the activation to calc the y<t>
        Returns
        -------
        Nothing.
        """
        pass
