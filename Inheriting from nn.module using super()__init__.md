The __init__ method of torch.nn.Module in PyTorch is not empty, but it does not require any arguments during initialization because it initializes internal attributes necessary for managing the layers, parameters, and forward operations of the neural network model.

When you create a custom model in PyTorch by subclassing nn.Module, you typically call super().__init__() in your custom model's constructor (__init__) to ensure the parent class's initialization logic runs.

Why don't we pass arguments to super().__init__()?

The nn.Module class handles most of its setup internally and does not require external inputs to do so. It sets up foundational elements, such as:

A dictionary to store submodules (_modules).

A dictionary for parameters (_parameters).

Buffers, hooks, and other structural components required for building a model.


These are generic components that every PyTorch model needs, so PyTorch automatically handles them without requiring user input.

Example:

Here's an example of a custom PyTorch model:

import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()  # Calls nn.Module's __init__
        self.linear = nn.Linear(input_size, output_size)  # Define a linear layer
    
    def forward(self, x):
        return self.linear(x)  # Forward pass through the linear layer

# Instantiate the model
model = CustomModel(input_size=10, output_size=1)
print(model)

What happens in nn.Module.__init__?

When you call super().__init__() in the above code:

1. It sets up internal structures such as _modules, _parameters, _buffers, etc., which the PyTorch framework needs to manage the model's components.


2. Allows registering layers (e.g., nn.Linear) and ensures they are properly added to the _modules dictionary for seamless handling of operations like saving, loading, or transferring the model to a device (e.g., GPU).



Output of the code:

CustomModel(
  (linear): Linear(in_features=10, out_features=1, bias=True)
)

Key Points:

nn.Module is designed to handle initialization generically for all models, so you don’t need to pass anything to super().__init__().

Any specific layers or parameters that your model requires should be initialized in your custom model’s __init__ method.

Calling super().__init__() ensures that the base class's initialization logic runs, enabling the proper functioning of the framework features such as automatic parameter registration.


