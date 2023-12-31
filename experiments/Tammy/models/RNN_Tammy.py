import torch
import torch.nn as nn
import torch.nn.init


class RNNCell(nn.Module):
    """
    Single cell of RNN model
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str,
        bias: bool = True,
    ) -> None:
        """
        Initialize recurrent neural network
        Parameters
        --------
            input_size: int
                Number of feature in the input x
            output_size: int
                Number of feature in the output y
            hidden_size: int
                Number of feature in the hidden state h
            bias: bool
                Whether to include a bias term in the linear transformations
            activation: str
                Activation function to apply to the hidden state, there are 2
                options: tanh and relu
        Returns
        -------
        nothing
        """
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        if activation not in ["tanh", "relu"]:
            raise ValueError("Invalid activation function")
        self.activation = activation
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.init_parameters()

    def forward(self, input: torch.Tensor, hs_pre: torch.Tensor = None):
        """
        Computes the forward propagation of the RNN cell
        Parameters
        --------
            input: torch.Tensor
                Input tensor of shape (batch_size, input_size).
            hs_pre: torch.Tensor
                Previous hidden state tensor of shape (batch_size, hidden_size)
                Default is None, the initial hidden state is set to zeros.
        Returns
        -------
            hs: torch.Tensor
                Output hidden state tensor of shape (batch_size, hidden_size).
        """
        if hs_pre is None:
            hs_pre = torch.zeros(input.size(0), self.hidden_size)
        hs = self.x2h(input) + self.h2h(hs_pre)
        if self.activation == "tanh":
            hs = torch.tanh(hs)
        else:
            hs = torch.relu(hs)
        return hs

    def init_parameters(self) -> None:
        """
        Initialize the weights and biases of the RNN cell
        followed by Xavier normalization
        Parameters
        --------
            None
        Returns
        -------
            None
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_uniform_(param)
            if "bias" in name:
                param = param.view(1, param.size(0))
                torch.nn.init.xavier_uniform_(param)


class RNN(nn.Module):
    """
    Implement RNN model
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        activation: str,
        bias: bool = True,
    ) -> None:
        """
        Recurrent Neural Network (RNN) model.
        Parameters
        --------
            input_size: int
                Number of features in the input x
            hidden_size: int
                Number of features in the hidden state h
            output_size: int
                Number of features in the output y
            num_layers: int
                Number of RNN cell layers
            bias: bool
                Whether to include a bias term in the linear transformations
            activation: str
                Activation function to apply to the hidden state,
                there are 2 options: tanh and relu
        Returns
        --------
        None
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        if activation not in ["tanh", "relu"]:
            raise ValueError("Invalid activation function")
        self.fc = nn.Linear(hidden_size, output_size, bias=self.bias)
        self.rnn_cell = RNNCell(input_size, hidden_size, activation, num_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward propagation of the RNN model
        Parameters
        --------
            input: torch.Tensor
                Input tensor of shape (batch_size, sequence_length, input_size)
        Returns
            out: torch.Tensor
                Output tensor of shape (batch_size, output_size)
        """
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        output = []
        hn = h0[0, :, :]
        for seq in range(input.size(1)):
            hn = self.rnn_cell(input[:, seq, :], hn)
            output.append(hn)
        out = output[-1].squeeze()
        out = self.fc(out)
        return out
