# Import libs and frameworks
import logging

import torch
import torch.nn as nn

import experiments.hoang.models.model_utils as model_utils

# Logger
logging.basicConfig(format="%(asctime)s %(message)s")
log = logging.getLogger("test_logger")
log.setLevel(logging.DEBUG)


class RNN(nn.Module):
    """
    The (one to one) RNN class.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        bias: bool = False,
        activation1: str = "sigmoid",
        activation2: str = "sigmoid",
    ) -> None:
        """
        Constructor of the class.
        Parameters
        ----------
        input_size: embedding size of the input
        hidden_size: hidden size
        output_size: embedding size of the output
        bias: set to "True" to enable bias
        activation1: set the activation to calc the a<t>
        activation2: set the activation to calc the y<t>
        Returns
        -------
        Nothing.
        """
        if (
            type(input_size) != int
            or type(hidden_size) != int
            or type(output_size) != int
            or type(activation1) != str
            or type(activation2) != str
        ):
            raise TypeError("Wrong input type!")
        if any(x <= 0 for x in [input_size, hidden_size, output_size]):
            log.error(
                "Negative or zero number(s) is/are passed in the parameters. "
                "Please review and re-init the object."
            )
            raise ValueError(
                "Negative number detected in the parameters. Please review and"
                " re-init the object."
            )
        if type(bias) != bool:
            log.error(
                "Bias parameter is not bool variable. Please review and"
                " re-init the object."
            )
            raise TypeError(
                "Bias parameter is not bool variable. Please review and"
                " re-init the object."
            )
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.aa = nn.Linear(hidden_size, hidden_size)
        self.ax = nn.Linear(input_size, hidden_size)
        self.ya = nn.Linear(hidden_size, output_size)
        self.a = torch.zeros(1, hidden_size)
        self.bias = bias
        if bias:
            self.ba = nn.Parameter(torch.rand(1, hidden_size))
            self.by = nn.Parameter(torch.rand(1, output_size))
        self.activation1 = model_utils.define_activation(activation1)
        self.activation2 = model_utils.define_activation(activation2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the class.
        Parameters
        ----------
        x: The input
        Returns
        -------
        y_t: The result of RNN.
        """
        #         print("X shape: ", x.clone().detach().shape)
        if x.clone().detach().shape[1] != self.input_size:
            log.error("Wrong input size!")
            return
        if not self.bias:
            a_t = self.activation1(self.aa(self.a) + self.ax(x))
            self.a = a_t
            y_t = self.activation2(self.ya(a_t))
        else:
            a_t = self.activation1(self.aa(self.a) + self.ax(x) + self.ba)
            self.a = a_t
            y_t = self.activation2(self.ya(a_t) + self.by)
        #         print(a_t.shape)
        return y_t

    def reset_a(self) -> None:
        """
        Reset self.a to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.a = torch.zeros(1, self.hidden_size)


class manyToOneRNN(nn.Module):
    """
    Many to one RNN class.
    """

    def __init__(
        self,
        input_times: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        bias: bool = False,
        activation1: str = "sigmoid",
        activation2: str = "sigmoid",
    ) -> None:
        """
        Constructor of the class.
        Parameters
        ----------
        input_times: Times of inputs
        other parameters: Same with RNN class
        Returns
        -------
        Nothing
        """
        if type(input_times) != int:
            raise TypeError("input times must be int")
        if input_times < 1:
            log.error("Input times less than 1")
            raise ValueError("Input times less than 1")
        super(manyToOneRNN, self).__init__()
        self.rnn = RNN(
            input_size,
            hidden_size,
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
        for i in range(x.shape[1]):
            y_t = self.rnn.forward(x[:, i, :])
        self.reset_a()
        return y_t

    def reset_a(self) -> None:
        """
        Reset self.a of self.rnn to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.rnn.reset_a()


class oneToManyRNN(nn.Module):
    """
    The one to many RNN class.
    """

    def __init__(
        self,
        output_times: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        bias: bool = False,
        activation1: str = "sigmoid",
        activation2: str = "sigmoid",
    ) -> None:
        """
        Constructor of the class.
        Parameters
        ----------
        output_times: Times of outputs
        other parameters: Same with RNN class
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
        super(oneToManyRNN, self).__init__()
        self.rnn = RNN(
            input_size,
            hidden_size,
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
        y_t = self.rnn.forward(x)
        result = torch.cat((result, y_t), 0)
        #         print(result)
        for _ in range(self.output_times - 1):
            y_t = self.rnn.forward(y_t)
            result = torch.cat((result, y_t), 0)
        #             print(y_t)
        self.reset_a()
        return torch.reshape(
            result, (x.shape[0], self.output_times, self.rnn.output_size)
        )  # Batch size, output times, output size

    def reset_a(self) -> None:
        """
        Reset self.a of self.rnn to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.rnn.reset_a()


class manyToManyRNN(nn.Module):
    """
    Many to many RNN class.
    """

    def __init__(
        self,
        input_times: int,
        output_times: int,
        input_size: int,
        hidden_size: int,
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
        simultaneous: Choose if RNN receive all the inputs before
        other parameters: Same with RNN class.
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
            log.error("Either input times or output times < 1")
            raise ValueError("Either input times or output times < 1")
        if input_size != output_size and simultaneous is False:
            log.error("Input size and output size is different.")
            raise ValueError("Input size and output size is different.")
        if simultaneous and input_times != output_times:
            raise ValueError("If simultaneous, input times must equal output times")
        super(manyToManyRNN, self).__init__()
        self.rnn = RNN(
            input_size,
            hidden_size,
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
        if len(x.shape) != 2 and len(x.shape) != 3:
            log.error("Wrong input size!")
            raise ValueError("Wrong input size!")
        result = torch.tensor([])
        if self.simultaneous:
            for i in range(x.shape[1]):
                y_t = self.rnn.forward(x[:, i, :])
                result = torch.cat((result, y_t), 0)
        else:
            y_t = 0
            for i in range(x.shape[1]):
                y_t = self.rnn.forward(x[:, i, :])

            for _ in range(self.output_times):
                y_t = self.rnn.forward(y_t)
                result = torch.cat((result, y_t), 0)
        self.reset_a()
        return torch.reshape(
            result, (x.shape[0], self.output_times, self.rnn.output_size)
        )  # Batch size, output times, output size

    def reset_a(self) -> None:
        """
        Reset self.a of self.rnn to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.rnn.reset_a()


class manyToManyBidRNN(nn.Module):
    """
    Many to many Bidirectional RNN class
    """

    def __init__(
        self,
        input_times: int,
        output_times: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        bias: bool = False,
        activation1: str = "tanh",
        activation2: str = "tanh",
    ):
        """
        Constructor of the class
        Parameters
        ----------
        input_times: Times of the input
        output_times: Times of the output
        simultaneous: Choose if RNN receive all the inputs before
        other parameters: Same with RNN class.
        Returns
        -------
        Nothing
        """
        if (
            type(output_times) != int
            or type(input_times) != int
            or type(input_size) != int
            or type(output_size) != int
        ):
            raise TypeError("type(s) of parameters passed is/are wrong.")
        if output_times < 1 or input_times < 1:
            log.error("Either input times or output times < 1")
            raise ValueError("Either input times or output times < 1")
        if output_times != input_times:
            raise ValueError("output times not equal input times")
        super(manyToManyBidRNN, self).__init__()
        self.rnn = RNN(
            input_size,
            hidden_size,
            output_size,
            bias,
            activation1,
            activation2,
        )
        self.input_times = input_times
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
        if len(x.shape) != 2 and len(x.shape) != 3:
            log.error("Wrong input size!")
            raise ValueError("Wrong input size!")
        result = torch.tensor([])

        if len(x.shape) == 3:
            length = x.shape[1]
            for i in range(length):
                y_t = self.rnn.forward(x[:, i, :])
            for i in range(length):
                y_t = self.rnn.forward(x[:, length - i - 1, :])
                result = torch.cat((result, y_t), 0)
            self.reset_a()
            return torch.reshape(
                result, (x.shape[0], self.output_times, self.rnn.output_size)
            )  # Batch size, output times, output size
        else:
            length = x.shape[0]
            for i in range(length):
                y_t = self.rnn.forward(x[i, :].unsqueeze(0))
            for i in range(length):
                y_t = self.rnn.forward(x[length - i - 1, :].unsqueeze(0))
                result = torch.cat((result, y_t), 0)
            self.reset_a()
            return torch.reshape(
                result, (self.output_times, self.rnn.output_size)
            )  # Batch size, output times, output size

    def reset_a(self):
        """
        Reset self.a of self.rnn to avoid error.
        Parameters
        ----------
        Nothing
        Returns
        -------
        Nothing
        """
        self.rnn.reset_a()


if __name__ == "__main__":
    testClass = manyToManyBidRNN(4, 4, 5, 512, 20, True)
    print(testClass.forward(torch.rand(4, 5)).shape)
    print(testClass.forward(torch.rand(4, 5)))

    log.info("nothing here!")
