# Import libs and frameworks
import logging

import torch
import torch.nn as nn

import experiments.hoang.models.model_utils as model_utils

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
        # Check condition
        if (
            type(input_size) != int
            or type(output_size) != int
            or type(bias) != bool
            or type(activation1) != str
            or type(activation2) != str
        ):
            raise TypeError("Type(s) of parameter(s) is wrong.")

        if input_size < 1 or output_size < 1:
            raise ValueError("Input size and/or output size can't below 1")

        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias

        # Init weights
        self.w_f = nn.Linear(input_size, output_size)
        self.u_f = nn.Linear(output_size, output_size)
        self.w_i = nn.Linear(input_size, output_size)
        self.u_i = nn.Linear(output_size, output_size)
        self.w_o = nn.Linear(input_size, output_size)
        self.u_o = nn.Linear(output_size, output_size)
        self.w_c = nn.Linear(input_size, output_size)
        self.u_c = nn.Linear(output_size, output_size)
        self.activation1 = model_utils.define_activation(activation1)
        self.activation2 = model_utils.define_activation(activation2)
        self.h_t = torch.zeros(1, output_size)
        self.c_t = torch.zeros(1, output_size)

        # Bias weights
        if bias:
            self.b_f = nn.Parameter(torch.rand(1, output_size))
            self.b_i = nn.Parameter(torch.rand(1, output_size))
            self.b_o = nn.Parameter(torch.rand(1, output_size))
            self.b_c = nn.Parameter(torch.rand(1, output_size))
        else:
            self.b_f = torch.zeros(1, output_size)
            self.b_i = torch.zeros(1, output_size)
            self.b_o = torch.zeros(1, output_size)
            self.b_c = torch.zeros(1, output_size)

    def forward(self, x: torch.tensor, h_t: torch.tensor = None) -> torch.tensor:
        """
        Forward function of the class
        Parameters
        ----------
        x: The input
        h_t: Use if user want something crazy. Better to leave it as it be
        Returns
        -------
        y_t: The result of RNN
        """
        if x.clone().detach().shape[1] != self.input_size:
            log.error("Wrong input size!")
            return
        if h_t is not None:
            self.h_t = h_t
        f_t = self.activation1(self.w_f(x) + self.u_f(self.h_t) + self.b_f)
        i_t = self.activation1(self.w_i(x) + self.u_i(self.h_t) + self.b_i)
        o_t = self.activation1(self.w_o(x) + self.u_o(self.h_t) + self.b_o)
        c_tilde_t = self.activation2(self.w_c(x) + self.u_c(self.h_t) + self.b_c)
        self.c_t = f_t * self.c_t + i_t * c_tilde_t
        self.h_t = o_t * self.activation2(self.c_t)
        return self.h_t

    def reset_states(self) -> None:
        """
        Reset the states in the model
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.c_t = torch.zeros(1, self.output_size)
        self.h_t = torch.zeros(1, self.output_size)


class manyToOneLSTM(nn.Module):
    """
    Many to one LSTM class.
    """

    def __init__(
        self,
        input_times: int,
        input_size: int,
        output_size: int,
        bias: bool = False,
        activation1: str = "sigmoid",
        activation2: str = "tanh",
    ) -> None:
        """
        Constructor of the class.
        Parameters
        ----------
        input_times: Times of inputs
        other parameters: Same with LSTM class
        Returns
        -------
        Nothing
        """
        if type(input_times) != int:
            raise TypeError("input times must be int")
        if input_times < 1:
            log.error("Input times less than 1")
            raise ValueError("Input times less than 1")
        super(manyToOneLSTM, self).__init__()
        self.lstm = LSTM(
            input_size,
            output_size,
            bias,
            activation1,
            activation2,
        )
        self.input_times = input_times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of this class
        Parameters
        ----------
        x: Input
        Returns
        -------
        Nothing
        """
        if x.clone().detach().shape[1] != self.input_times:
            log.error("Wrong input size!")
            raise ValueError("Wrong input size!")
        y_t = None
        for i in range(x.shape[1]):
            y_t = self.lstm.forward(x[:, i, :])
        self.reset_states()
        return y_t

    def reset_states(self) -> None:
        """
        Reset the states to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.lstm.reset_states()


class oneToManyLSTM(nn.Module):
    """
    The one to many GRU class.
    """

    def __init__(
        self,
        output_times: int,
        input_size: int,
        output_size: int,
        bias: bool = False,
        activation1: str = "sigmoid",
        activation2: str = "tanh",
    ) -> None:
        """
        Constructor of the class.
        Parameters
        ----------
        output_times: Times of outputs
        other parameters: Same with LSTM class
        Returns
        -------
        Nothing
        """
        if type(output_times) != int:
            raise TypeError("Output times is not int")
        if output_times < 1:
            log.error("Output times < 1.")
            raise ValueError("Output times < 1.")
        if type(input_size) != int or type(output_size) != int:
            raise TypeError("Wrong type of input size or output size")
        if input_size != output_size:
            log.error("Input size and output size is different")
            raise ValueError("Input size and output size is different")
        super(oneToManyLSTM, self).__init__()
        self.lstm = LSTM(
            input_size,
            output_size,
            bias,
            activation1,
            activation2,
        )
        self.output_times = output_times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of this class
        Parameters
        ----------
        x: Input
        Returns
        -------
        Nothing
        """
        if len(x.clone().detach().shape) != 3:
            log.error("Wrong input size!")
            return
        result = torch.Tensor([])
        y_t = self.lstm.forward(x)
        result = torch.cat((result, y_t), 0)
        #         print(result)
        for _ in range(self.output_times - 1):
            y_t = self.lstm.forward(y_t)
            result = torch.cat((result, y_t), 0)
        #             print(y_t)
        self.reset_states()
        return torch.reshape(
            result, (x.shape[0], self.output_times, self.lstm.output_size)
        )  # Batch size, output times, output size

    def reset_states(self) -> None:
        """
        Reset the states to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.lstm.reset_states()


class manyToManyLSTM(nn.Module):
    """
    Many to many LSTM class.
    """

    def __init__(
        self,
        input_times: int,
        output_times: int,
        input_size: int,
        output_size: int,
        bias: bool = False,
        activation1: str = "sigmoid",
        activation2: str = "sigmoid",
        simultaneous: bool = False,
    ) -> None:
        """
        Constructor of the class
        Parameters
        ----------
        input_times: Times of the input
        output_times: Times of the output
        simultaneous: Choose if LSTM receive all the inputs before
        other parameters: Same with LSTM class.
        Returns
        -------
        Nothing
        """
        if (
            type(output_times) != int
            or type(input_times) != int
            or type(input_size) != int
            or type(output_size) != int
            or type(simultaneous) != bool
        ):
            raise TypeError("type(s) of parameters passed is/are wrong.")
        if output_times < 1 or input_times < 1:
            log.error(
                "Either input times or output times < 1. It will cause errors"
                " in the future. Please re-init the object."
            )
            raise ValueError(
                "Either input times or output times < 1. It will cause errors"
                " in the future. Please re-init the object."
            )
        if input_size != output_size and simultaneous is False:
            log.error("Input size and output size is different.")
            raise ValueError("Input size and output size is different.")
        if simultaneous and input_times != output_times:
            raise ValueError("If simultaneous, input times must equal output times")
        super(manyToManyLSTM, self).__init__()
        self.lstm = LSTM(
            input_size,
            output_size,
            bias,
            activation1,
            activation2,
        )
        self.input_times = input_times
        self.output_times = output_times
        self.simultaneous = simultaneous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of this class
        Parameters
        ----------
        x: Input
        Returns
        -------
        Nothing
        """
        if len(x.shape) != 3:
            log.error("Wrong input size!")
            raise ValueError("Wrong input size!")
        result = torch.tensor([])
        if self.simultaneous:
            for i in range(x.shape[1]):
                y_t = self.lstm.forward(x[:, i, :])
                result = torch.cat((result, y_t), 0)
        else:
            y_t = 0
            for i in range(x.shape[1]):
                y_t = self.lstm.forward(x[:, i, :])

            for _ in range(self.output_times):
                y_t = self.lstm.forward(y_t)
                result = torch.cat((result, y_t), 0)
        self.reset_states()
        return torch.reshape(
            result, (x.shape[0], self.output_times, self.lstm.output_size)
        )  # Batch size, output times, output size

    def reset_states(self) -> None:
        """
        Reset the states to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.lstm.reset_states()
