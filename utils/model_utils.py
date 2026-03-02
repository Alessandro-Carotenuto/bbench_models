import torch.nn as nn

def getActivation(f_identifier : str):
    f_identifier=f_identifier.lower()

    if f_identifier in ['softmax', 'log_softmax']:
        if f_identifier == 'softmax': 
            return nn.Softmax(dim=1)
        return nn.LogSoftmax(dim=1)

    activations = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'gelu': nn.GELU,
            'elu': nn.ELU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'none': nn.Identity
        }
    
    if f_identifier not in activations:
        raise ValueError(f"Activation '{f_identifier}' is not supported.")

    activation_class = activations[f_identifier]

    
    return activation_class()
