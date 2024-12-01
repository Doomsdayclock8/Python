# When you call loss.backward(), the gradient of the loss is computed only with respect to the variables (or parameters) that have requires_grad=True
### Key Points:

1. **`requires_grad=True` Marks Variables for Gradient Calculation:**
    
    - PyTorch tracks operations performed on tensors with `requires_grad=True`.
    - During backpropagation, gradients are computed only for those tensors because only they are part of the computational graph used to compute the loss.
2. **Other Variables Are Ignored:**
    
    - Tensors with `requires_grad=False` are treated as constants, even if they are involved in the computation of the loss.
    - Their contribution to the loss is accounted for, but PyTorch does not compute gradients for them.
3. **Why This Behavior?**
    
    - It is an efficiency feature: PyTorch avoids unnecessary computation and memory usage by not computing gradients for variables that don't require them.
    - Typically, model weights and biases have `requires_grad=True`, while input data and other constants do not.
```
import torch

# Define tensors
x = torch.tensor(2.0, requires_grad=True)  # Gradient will be computed for this
y = torch.tensor(3.0, requires_grad=False)  # Treated as a constant, no gradient
z = x * y  # z = 2.0 * 3.0 = 6.0
loss = z ** 2  # loss = (6.0)^2 = 36.0

# Backward pass
loss.backward()

# Check gradients
print("Gradient of x:", x.grad)  # Output: 36.0 * 2 * 3 = 216.0
print("Gradient of y:", y.grad)  # Output: None, because y.requires_grad=False

```
### **Explanation:**

- The gradient of `loss` with respect to `x` is computed: ∂(loss)∂x=2z⋅∂z∂x=2(6)⋅3=216\frac{\partial (\text{loss})}{\partial x} = 2z \cdot \frac{\partial z}{\partial x} = 2(6) \cdot 3 = 216∂x∂(loss)​=2z⋅∂x∂z​=2(6)⋅3=216
- No gradient is computed for `y` because `requires_grad=False`.