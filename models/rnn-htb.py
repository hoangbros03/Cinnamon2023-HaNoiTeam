# Import libs and frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Logger
log = logging.getLogger("test_logger")

class RNN(nn.Module):
    """ 
    The (one to one) RNN class.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,bias: bool = False, activation1: str = "sigmoid", activation2: str = "sigmoid") -> None:
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
        if any(x < 0 for x in [input_size, hidden_size, output_size]):
            log.error("Negative number detected in the parameters. Please review and re-init the object.")
        if type(bias)!= bool:
            log.error("Bias parameter is not bool variable. Please review and re-init the object.")
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
            self.ba = torch.rand(1,hidden_size)
            self.by = torch.rand(1, output_size)
        self.activation1 = self.define_activation(activation1)
        self.activation2 = self.define_activation(activation2)
        
    def define_activation(self, typeActivation: str) -> None:
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
        
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
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
            a_t = self.activation1(self.aa(self.axa) + self.ax(x)+self.ba)
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
    def __init__(self, input_times: int, input_size: int, hidden_size: int, output_size: int, bias: bool = False, activation1: str = "sigmoid", activation2: str = "sigmoid") -> None:
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
        if input_times < 1: 
            log.error("Input times less than 1. This object stills be created but won't work!")
        super(manyToOneRNN, self).__init__()
        self.rnn = RNN(input_size, hidden_size, output_size, bias, activation1, activation2)
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
            return
        for i in range(x.shape[1]):
            y_t = self.rnn.forward(x[:,i,:])
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
    def __init__(self, output_times: int, input_size: int, hidden_size: int, output_size: int, bias: bool = False, activation1: str = "sigmoid", activation2: str = "sigmoid") -> None:
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
        if output_times < 1:
            log.error("Output times < 1. It will cause errors in the future. Please re-init the object.")
        if input_size != output_size:
            log.error("Input size and output size is different. This object stills be created but won't work!")
        super(oneToManyRNN, self).__init__()
        self.rnn = RNN(input_size, hidden_size, output_size, bias, activation1, activation2)
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
        result = torch.cat((result, y_t),0)
#         print(result)
        for i in range(self.output_times - 1):
            y_t = self.rnn.forward(y_t)
            result = torch.cat((result,y_t),0)
#             print(y_t)
        self.reset_a()
        return torch.reshape(result,(x.shape[0], self.output_times, self.rnn.output_size)) # Batch size, output times, output size
    
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
    def __init__(self, input_times: int, output_times: int, input_size: int, hidden_size: int, output_size: int, bias: bool = False, activation1: str = "sigmoid", activation2: str = "sigmoid", simultaneous: bool = False) -> None:
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
        if output_times < 1 or input_times < 1:
            log.error("Either input times or output times < 1. It will cause errors in the future. Please re-init the object.")
        if input_size != output_size:
            log.error("Input size and output size is different. This object stills be created but won't work!")
        super(manyToManyRNN, self).__init__()
        self.rnn = RNN(input_size, hidden_size, output_size, bias, activation1, activation2)
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
        result = torch.tensor([])
        if self.simultaneous:
            for i in range(x.shape[1]):

                y_t = self.rnn.forward(x[:,i,:])
                result = torch.cat((result, y_t),0)
        else:
            y_t = 0
            for i in range(x.shape[1]):
      
                y_t = self.rnn.forward(x[:,i,:])

            for i in range(self.output_times):
           
                y_t = self.rnn.forward(y_t)
                result = torch.cat((result, y_t),0)
        self.reset_a()
        return torch.reshape(result,(x.shape[0], self.output_times, self.rnn.output_size)) # Batch size, output times, output size
    
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