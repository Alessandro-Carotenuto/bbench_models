import torch.nn as nn
from typing import List
from utils.model_utils import getActivation

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
        
        self.layer_activation=getActivation(activation_per_layer)
        self.output_activation=getActivation(activation_output)
        

    def forward(self, x):
        for i in range(0,self.num_layers):
            x=self.MLPlayers[i](x)
            if i < self.num_layers-1:
                x=self.layer_activation(x)
            else:
                x=self.output_activation(x)
        return x

