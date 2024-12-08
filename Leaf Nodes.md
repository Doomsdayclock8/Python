- A **leaf node** is a tensor that is directly created by the user and has `requires_grad=True`. For example, weights and biases initialized in a model are leaf nodes.
- A **non-leaf node** is the result of an operation involving other tensors, and it typically will not have `requires_grad=True` by default unless its inputs also require gradients. These nodes do not hold the gradient information themselves, but instead, their gradients will be stored in the parent (leaf) nodes during backpropagation.
#### Example:

```python
import torch

# Creating a tensor directly with requires_grad=True
x = torch.randn(3, 3, requires_grad=True)  # This is a leaf node

# Performing an operation
y = x * 2  # This is not a leaf node

# Checking if they are leaf nodes
print(x.is_leaf)  # True
print(y.is_leaf)  # False
```

In this example, `x` is a leaf node because it was created directly with `requires_grad=True`, while `y` is not a leaf node since it's the result of an operation on `x`.