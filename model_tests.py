import torch
from models.mlp import Multi_Layer_Perceptron

def MLP_class_test():

    input_dim=10
    output_dim=10
    hidden_dims=[10,10]
    activation_per_layer='relu'
    activation_output='softmax'


    model=Multi_Layer_Perceptron(
        input_dim,
        output_dim,
        hidden_dims,
        activation_per_layer,
        activation_output
    )

    print(" -> Model Created")

    batch_size = 5
    dummy_input = torch.randn(batch_size,input_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, output_dim), f"Wrong output shape: {output.shape}"
    print(" -> Shape Test Passed")


if __name__ == "__main__":
    MLP_class_test()
