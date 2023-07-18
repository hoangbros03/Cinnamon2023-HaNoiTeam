# Import libs and frameworks
import logging

import torch
import torch.nn as nn

import experiments.hoang.models.model_utils as model_utils

# Logger
logging.basicConfig(format="%(asctime)s %(message)s")
log = logging.getLogger("test_logger")
log.setLevel(logging.DEBUG)


class GRU(nn.Module):
    """
    The (a) Gated recurrent unit class.
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
        # Check errors
        if (
            type(input_size) != int
            or type(output_size) != int
            or type(bias) != bool
            or type(activation1) != str
            or type(activation2) != str
        ):
            raise TypeError("Type(s) of parameter(s) is wrong.")

        super(GRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias

        # Init weights
        self.w_z = nn.Linear(input_size, output_size)
        self.u_z = nn.Linear(output_size, output_size)
        self.w_r = nn.Linear(input_size, output_size)
        self.u_r = nn.Linear(output_size, output_size)
        self.w_h = nn.Linear(input_size, output_size)
        self.u_h = nn.Linear(output_size, output_size)
        self.h_t = torch.zeros(1, output_size)
        if bias:
            self.b_z = nn.Parameter(torch.rand(1, self.output_size))
            self.b_r = nn.Parameter(torch.rand(1, self.output_size))
            self.b_h = nn.Parameter(torch.rand(1, self.output_size))
        else:
            self.b_z = torch.zeros(1, self.output_size)
            self.b_r = torch.zeros(1, self.output_size)
            self.b_h = torch.zeros(1, self.output_size)
        self.activation1 = model_utils.define_activation(activation1)
        self.activation2 = model_utils.define_activation(activation2)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward function of the class
        Parameters
        ----------
        x: The input
        Returns
        -------
        y_t: The result of RNN
        """
        if x.clone().detach().shape[1] != self.input_size:
            log.error("Wrong input size!")
            return
        z_t = self.activation1(self.w_z(x) + self.u_z(self.h_t) + self.b_z)
        r_t = self.activation1(self.w_r(x) + self.u_r(self.h_t) + self.b_r)
        h_hat_t = self.activation2(self.w_h(x) + self.u_h(r_t * self.h_t) + self.b_h)
        self.h_t = (1 - z_t) * self.h_t + z_t * h_hat_t
        return self.h_t

    def reset_h_t(self) -> None:
        """
        Reset self.a to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.h_t = torch.zeros(1, self.output_size)


if __name__ == "__main__":
    gru = GRU(3, 3, False)
    result = gru.forward(torch.rand(3, 3))
    print(result)
    print(result.shape)
