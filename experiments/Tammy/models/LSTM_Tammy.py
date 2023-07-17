import torch
import torch.nn as nn
import torch.nn.init


class LSTMCell(nn.Module):
    """
    Single cell of LSTM
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize long short term memory cell
        Parameters
        --------
          input_size: int
            The number of expected features in the input x
          hidden_size: int
            The number of features in the hidden state h
          bias: bool
            Optional, if False, the layer doesn't use bias weights b_ih and b_hh
            Default: True
        Returns
        -------
        None

        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # input gate (i)
        self.input_x2i = nn.Linear(input_size, hidden_size, bias=bias)
        self.input_h2i = nn.Linear(hidden_size, hidden_size, bias=bias)

        # forgot gate (f)
        self.forgot_x2f = nn.Linear(input_size, hidden_size, bias=bias)
        self.forgot_h2f = nn.Linear(hidden_size, hidden_size, bias=bias)

        # cell vector (c)
        self.cell_x2c = nn.Linear(input_size, hidden_size, bias=bias)
        self.cell_h2c = nn.Linear(hidden_size, hidden_size, bias=bias)

        # almost output (o)
        self.output_x2o = nn.Linear(input_size, hidden_size, bias=bias)
        self.output_h20 = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.init_parameters()

    def input_gate(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: size (batch_size, input_size)
        h: size (batch_size, hidden_size)
        i: size (batch_size, hidden_size)
        """
        x_t = self.input_x2i(x)
        hs_pre = self.input_h2i(h)
        acti = nn.Sigmoid()
        i = acti(x_t + hs_pre)
        return i

    def forgot_gate(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: size (batch_size, input_size)
        h: size (batch_size, hidden_size)
        f: size (batch_size, hidden_size)
        """
        x_t = self.forgot_x2f(x)
        hs_pre = self.forgot_h2f(h)
        acti = nn.Sigmoid()
        f = acti(x_t + hs_pre)
        return f

    def cell_vector(
        self,
        i: torch.Tensor,
        f: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor,
        c_pre: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: size (batch_size, input_size)
        h: size (batch_size, hidden_size)
        c: size (batch_size, hidden_size)
        """
        x_t = self.cell_x2c(x)
        hs_pre = self.cell_h2c(h)
        acti = nn.Tanh()
        c = f * c_pre + i * acti(x_t + hs_pre)
        return c

    def almost_output(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: size (batch_size, input_size)
        h: size (batch_size, hidden_size)
        out: size (batch_size, hidden_size)
        """
        x_t = self.output_x2o(x)
        hs_pre = self.output_h20(h)
        acti = nn.Sigmoid()
        out = acti(x_t + hs_pre)
        return out

    def forward(
        self, x: torch.Tensor, h_n_c: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward propagation of the LSTM cell
        Parameters
        --------
            x: torch.Tensor
                Input tensor of shape (batch_size, input_size).
            h_n_c: tuple[torch.Tensor, torch.Tensor]
                Previous hidden state tensor of shape
                ((batch_size, hidden_size), (batch_size, hidden_size))
                Default is None, the initial hidden state is set to zeros.
        Returns
        -------
            hs: tuple[torch.Tensor, torch.Tensor]
                Output hidden state tensor of shape
                ((batch_size, hidden_size), (batch_size, hidden_size)).
        """
        if h_n_c is None:
            h_n_c = torch.zeros(x.size(0), self.hidden_size)
            h_n_c = (h_n_c, h_n_c)
        (hs_pre, c_pre) = h_n_c
        i = self.input_gate(x, hs_pre)
        f = self.forgot_gate(x, hs_pre)
        c = self.cell_vector(i, f, x, hs_pre, c_pre)
        o = self.almost_output(x, hs_pre)
        acti = nn.Tanh()
        hs = o * acti(c)
        return (hs, c)

    def init_parameters(self) -> None:
        """
        Initialize the weights and biases of the LSTM cell
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


class LSTM(nn.Module):
    """
    Implements a multi-layer LSTM model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        bias: bool = True,
    ) -> None:
        """
        Initialize long short term memory
        Parameters
        --------
          input_size: int
            The number of expected features in the input x
          hidden_size: int
            The number of features in the hidden state h
          bias: bool
            Optional, if False, the layer doesn't use bias weights b_ih and b_hh.
            Default: True
        Returns
        -------
          None
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.fc = nn.Linear(self.hidden_size, self.output_size, bias)
        self.lstm_cell = LSTMCell(input_size, hidden_size, num_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.
        Parameters
        --------
          input: torch.Tensor
            The input tensor of shape (batch_size, sequence_length, input_size)
        Returns 
        --------
          output: torch.Tensor
            Output hidden state tensor of shape (batch_size, output_size)
        """
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        output = []
        cn = c0[0, :, :]
        hn = h0[0, :, :]
        for seq in range(input.size(1)):
            hn, cn = self.lstm_cell(input[:, seq, :], (hn, cn))
            output.append(hn)
        out = output[-1].squeeze()
        out = self.fc(out)
        return out
