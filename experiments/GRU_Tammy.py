import torch
import torch.nn as nn
import torch.nn.init


class GRUCell(nn.Module):
    """
    Single cell of GRU
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """
        Initialize gated recurrent unit cell
        Parameters
        --------
          input_size: int
            The number of expected features in the input x
          hidden_size: int
            The number of features in the hidden state h
          bias: bool
            Optional, if False,the layer doesn't use bias weights b_ih and b_hh
            Default: True
        Returns
        -------
            None

        """
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # reset gate (r)
        self.reset_i2r = nn.Linear(input_size, hidden_size, bias=bias)
        self.reset_h2r = nn.Linear(hidden_size, hidden_size, bias=bias)

        # update gate (z)
        self.update_i2z = nn.Linear(input_size, hidden_size, bias=bias)
        self.update_h2z = nn.Linear(hidden_size, hidden_size, bias=bias)

        # almost output (n)
        self.output_i2n = nn.Linear(input_size, hidden_size, bias=bias)
        self.output_h2n = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.init_parameters()

    def reset_gate(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: size (batch_size, input_size)
        h: size (batch_size, hidden_size)
        r: size (batch_size, hidden_size)
        """
        x_t = self.reset_i2r(x)
        hs_pre = self.reset_h2r(h)
        acti = nn.Sigmoid()
        r = acti(x_t + hs_pre)
        return r

    def update_gate(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: size (batch_size, input_size)
        h: size (batch_size, hidden_size)
        z: size (batch_size, hidden_size)
        """
        x_t = self.update_i2z(x)
        hs_pre = self.update_h2z(h)
        acti = nn.Sigmoid()
        z = acti(x_t + hs_pre)
        return z

    def almost_output(
        self, x: torch.Tensor, h: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:
        """
        x: size (batch_size, input_size)
        h: size (batch_size, hidden_size)
        r: size (batch_size, hidden_size)
        n: size (batch_size, hidden_size)
        """
        x_t = self.output_i2n(x)
        hs_pre = self.output_h2n(h)
        acti = nn.Tanh()
        n = acti(x_t + hs_pre)
        return n

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the forward propagation of the GRU cell
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
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size)
        r = self.reset_gate(x, h)
        z = self.update_gate(x, h)
        n = self.almost_output(x, h, r)
        hs = (1 - z) * n + z * h
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


class GRU(nn.Module):
    """
    Implements a multi-layer GRU model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        bias: bool = True,
    ) -> None:
        """
        Initialize gated recurrent unit
        Parameters
        --------
          input_size: int
            The number of expected features in the input x
          hidden_size: int
            The number of features in the hidden state h
          num_layers: int
            The number of layers
          output_size: int
            The number of output features
          bias: bool
            Optional, if False, then the layer does not use bias weights
            b_ih and b_hh. Default: True
        Returns
        -------
            None

        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bias = bias
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.init_cell_list()

    def forward(self, input: torch.Tensor, hs_pre: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the GRU model.
        Parameters
        --------
          input: torch.Tensor
            The input tensor of shape (batch_size, sequence_length, input_size)
          hs_pre: torch.Tensor
            Previous hidden state tensor of shape (batch_size, hidden_size)
            Default is None, the initial hidden state is set to zeros.
        Returns
        --------
          output: torch.Tensor
            Output hidden state tensor of shape (batch_size, hidden_size)
        """
        if hs_pre is None:
            hs_pre = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        output = []
        hidden_layers = list(hs_pre)
        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden = self.gru_cell_list[layer].forward(
                        input[:, t, :], hidden_layers[layer]
                    )
                else:
                    hidden = self.gru_cell_list[layer].forward(
                        hidden_layers[layer - 1], hidden_layers[layer]
                    )
                hidden_layers[layer] = hidden
            output.append(hidden)
        out = output[-1].squeeze()
        out = self.fc(out)
        return out

    def init_cell_list(self):
        """
        Initializes the GRU cell list based on the number of layers.
        """
        self.gru_cell_list = nn.ModuleList()
        self.gru_cell_list.append(GRUCell(self.input_size, self.hidden_size, self.bias))
        for _ in range(1, self.num_layers):
            self.gru_cell_list.append(
                GRUCell(self.hidden_size, self.hidden_size, self.bias)
            )
