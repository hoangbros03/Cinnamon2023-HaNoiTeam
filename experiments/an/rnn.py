import logging

import torch
import numpy as np

import csv

torch.manual_seed(1)

class RNNCell:
    '''
    RNN cell representing a single unit of RNN, which takes in input and a hidden 
    state and returns output and updated hidden state
    '''
    def __init__(self, input_size, hidden_size, output_size, batch_size, activation1, activation2):
        """
        Parameters:
        ...
        activation1, activation2: activation functions
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.activation1 = activation1
        self.activation2 = activation2
        self.initialize_paras()

    def initialize_paras(self):
        self.w1 = torch.rand(self.input_size, self.hidden_size)
        self.w2 = torch.rand(self.hidden_size, self.hidden_size)
        self.w3 = torch.rand(self.hidden_size, self.output_size)
        self.b1 = torch.rand(self.batch_size, self.hidden_size)
        self.b2 = torch.rand(self.batch_size, self.hidden_size)
        self.b3 = torch.rand(self.batch_size, self.output_size)

    def forward(self, x, prev_h):
        """
        Forward function of the class
        Parameters:
        -----------
        x: Input of shape batch_size * input_size
        prev_h: Previous hidden state of shape batch_size * hidden_size
        Returns:
        -----------
        Output of shape batch_size * output_size, hidden state of shape 
        batch_size * hidden_size
        """
        h = self.activation1.forward(
            torch.add(torch.add(torch.matmul(x, self.w1), self.b1), torch.add(torch.matmul(prev_h, self.w2), self.b2))
        )
        out = self.activation2.forward(torch.add(torch.matmul(h, self.w3), self.b3))
        return out, h


class RNNModel:
    """
    Many-to-many RNN model with the same input and output sequence length
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        batch_size: int = 1,
        activation1: str = "relu",
        activation2: str = "relu",
    ):
        """
        Constructor of the class

        Parameters:
        ------------
        input_size: number of features of the input
        hidden_size: number of features of the hidden layer
        output_size: number of features of the output
        batch_size: number of samples
        activation1: activation function to compute the hidden state at each timestep
        activation2: activation function to compute the output at each timestep
        Returns
        ------------
        Nothing
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.activation1 = self.get_activation(activation1)
        self.activation2 = self.get_activation(activation2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the class
        Parameters:
        -----------
        x: Input of shape timestep * batch_size * input_size
        Returns:
        -----------
        Output of shape timestep * batch_size * output_size
        """
        if x.size(2) != self.input_size:
            logging.error(f"Wrong input size. Input size expected: {self.input_size}")
        timestep = x.size(0)
        curr_h = torch.zeros(size=(self.batch_size, self.hidden_size))
        cell = RNNCell(self.input_size, self.hidden_size, self.output_size, self.batch_size, self.activation1, self.activation2)
        output = torch.zeros(size=(timestep, self.batch_size, self.output_size))
        for i in range(timestep):
            out, curr_h = cell.forward(x[i, :, :], curr_h)
            output[i, :, :] = out
        return output
    
    # def compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
    #     if pred.size() != target.size():
    #         logging.error("Wrong prediction value size")
    #     return torch.sqrt(pred - target)

    def get_activation(self, activation: str):
        """
        Get the activation function given the name
        Parameters:
        -----------
        activation: activation function name
        Returns:
        -----------
        activation function
        """
        if activation == "relu":
            return torch.nn.ReLU()
        elif activation == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation == "tanh":
            return torch.nn.Tanh()
        else:
            logging.error("Activation function not found. Return to default: ReLU")
            return torch.nn.ReLU()



rows = []
rows_output = []
with open('experiments/an/data.csv','r') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',',)
    for row in reader:
        rows.append(row[:19])
        rows_output.append(row[1:])
        
# print(rows)
X_train = torch.Tensor([rows])
Y_train = torch.Tensor([rows_output])
model = RNNModel(19,19,19,30, "relu", "relu")
# print(model.forward(X_train))
Y_pred = model.forward(X_train)
loss_func = torch.nn.MSELoss()
# loss = loss_func(Y_pred, Y_train)


optimizer = torch.optim.SGD(model.parameters())


# class ReLU:
#     """
#     Class to represent ReLU function
#     """

#     def forward(self, x):
#         """
#         Compute result of ReLU function
#         Parameters:
#         -----------
#         x: input -- tensor
#         Returns:
#         -----------
#         Output of the activation function -- tensor of the same size as input
#         """
#         return tf.keras.activations.relu(x)


# class Sigmoid:
#     """
#     Class to represent Sigmoid function
#     """

#     def forward(self, x):
#         """
#         Compute result of Sigmoid function
#         Parameters:
#         -----------
#         x: input -- tensor
#         Returns:
#         -----------
#         Output of the activation function -- tensor of the same size as input
#         """
#         return tf.keras.activations.sigmoid(x)


# class Tanh:
#     """
#     Class to represent Tanh function
#     """

#     def forward(self, x):
#         """
#         Compute result of Tanh function
#         Parameters:
#         -----------
#         x: input -- tensor
#         Returns:
#         -----------
#         Output of the activation function -- tensor of the same size as input
#         """
#         return tf.keras.activations.tanh(x)
