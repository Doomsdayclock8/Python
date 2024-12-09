### #More_on : [[Scalar vs Non-Scalar Gradient in Pytorch]]
The backward computation involves differentiating a scalar function with respect to its inputs (e.g., parameters xxx). For this to work:

1.  **Gradient Shape:**
	The gradient (∇f\nabla f∇f) should match the shape of the inputs.
    
    - For a scalar fff, ∇f\nabla f∇f is a tensor with the same shape as the inputs.
    - For a tensor fff (e.g., a vector or matrix), the gradient would require higher-order derivatives (a Jacobian matrix or Hessian tensor), which are not directly handled by `.backward()`.
2. **Practicality:**
	Most optimization problems in machine learning involve minimizing a **scalar loss function** (e.g., MSE, cross-entropy). Calling `.backward()` on a scalar automatically computes gradients for all parameters contributing to that scalar.
### How Gradients Work for Scalars

Let L be a scalar loss function:

- L=f(x)L = f(x)L=f(x), where xxx is a tensor of parameters.
- Calling `L.backward()` computes ∂L∂x\frac{\partial L}{\partial x}∂x∂L​, a tensor of the same shape as xxx.
### What if You Have a Non-Scalar Tensor?

For non-scalar tensors, you must reduce them to a scalar (e.g., sum, mean) before calling `.backward()`. This reduction defines how the gradients of individual elements contribute to the final gradient.
### Example 
```
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # Non-scalar tensor: y = [1, 4, 9]

# Reduce y to a scalar
z = y.sum()  # Scalar: z = 1 + 4 + 9 = 14
z.backward()  # Compute gradients

print(x.grad)  # Output: tensor([2., 4., 6.]) (dz/dx)

```
Here, the reduction to a scalar z=sum(y)z = \text{sum}(y)z=sum(y) ensures that gradients can be computed, as PyTorch knows how to backpropagate the sum operation.