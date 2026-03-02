import torch
import torch.nn as nn
from typing import List

class Multi_Layer_Perceptron(nn.Module):

    def __init__(
            self,
            input_dim : int,
            output_dim : int,
            hidden_dims : List[int],
            activation_per_layer : str = 'relu',
            activation_output: str = 'softmax'
        ):
        super().__init__()

        self.num_hidden_layers=len(hidden_dims)
        self.MLPlayers= nn.ModuleList()

        layer=nn.Linear(input_dim,hidden_dims[0])
        self.MLPlayers.append(layer)
        for i in range(0, self.num_hidden_layers-1):
            layer=nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.MLPlayers.append(layer)

        layer=nn.Linear(hidden_dims[self.num_hidden_layers-1],output_dim)
        self.MLPlayers.append(layer)
        self.num_layers=len(self.MLPlayers)
        
        self.layer_activation=self.getActivation(activation_per_layer)
        self.output_activation=self.getActivation(activation_output)
        

    def forward(self, x):
        for i in range(0,self.num_layers):
            x=self.MLPlayers[i](x)
            if i < self.num_layers-1:
                x=self.layer_activation(x)
            else:
                x=self.output_activation(x)
        return x

    @staticmethod
    def getActivation(f_identifier : str):
        f_identifier=f_identifier.lower()

        activations = {
                'relu': nn.ReLU,
                'leaky_relu': nn.LeakyReLU,
                'prelu': nn.PReLU,
                'gelu': nn.GELU,
                'elu': nn.ELU,
                'sigmoid': nn.Sigmoid,
                'tanh': nn.Tanh,
                'none': nn.Identity
            }
        
        if f_identifier not in activations:
            raise ValueError(f"Activation '{f_identifier}' is not supported.")
    
        activation_class = activations[f_identifier]

        if f_identifier in ['softmax', 'log_softmax']:
            if f_identifier == 'softmax': 
                return nn.Softmax(dim=1)
            return nn.LogSoftmax(dim=1)
        
        return activation_class()
