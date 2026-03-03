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
    dummy_input = torch.randn(batch_size,input_dim)               #CREATE one-single-batch of inputs 
    output = model(dummy_input)

    assert output.shape == (batch_size, output_dim), f"Wrong output shape: {output.shape}"
    print(" -> Shape Test Passed")
    softmax_sum=output.sum(dim=-1)

    assert torch.allclose(softmax_sum, torch.ones(batch_size), atol=1e-6), f"Wrong output sum: {softmax_sum}, expected around one"
    print(" -> Softmax Test Passed with 1e-6 tol")

    assert torch.isfinite(output).all(), "Outpu t contains NaN or Inf"
    print(" -> NaN/Inf Test Passed")

    fakeloss=output.sum() #number connected to computational graph
    fakeloss.backward()

    for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"
    print(" -> Gradient Test Passed")

    assert model.total_params > 0, "No parameters"
    assert model.trainable_params > 0, "No trainable parameters"
    print(" -> Params Test Passed")
    


if __name__ == "__main__":
    MLP_class_test()
