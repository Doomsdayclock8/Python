The **`unsqueeze(dim)`** operation in PyTorch adds a new dimension of size 1 at the specified position (`dim`) in the tensor's shape. It essentially increases the tensor's rank (number of dimensions) by one.

### **Why Use `unsqueeze`?**

- To align the dimensions of a tensor for broadcasting or compatibility with operations that require specific shapes.
- It does **not change the data** in the tensor, only its shape.

---

### **Example Usage**

#### Original Tensor:

```python
import torch
x = torch.tensor([1, 2, 3])  # Shape: [3]
print(x.shape)  # Output: torch.Size([3])
```

#### Adding a Dimension:

```python
x_unsqueezed = x.unsqueeze(1)  # Add a dimension at position 1
print(x_unsqueezed.shape)  # Output: torch.Size([3, 1])
```

#### Visual Representation:

If `x` is:

x=[1,2,3](Shape: [3])x = [1, 2, 3] \quad (\text{Shape: [3]})

Then `x.unsqueeze(1)` becomes:

x=[123](Shape: [3, 1])x = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \quad (\text{Shape: [3, 1]})

---

### **When to Use It?**

1. **Broadcasting**:
    
    ```python
    a = torch.tensor([1, 2, 3])  # Shape: [3]
    b = torch.tensor([[1], [2], [3]])  # Shape: [3, 1]
    
    # To perform element-wise addition:
    a = a.unsqueeze(1)  # Shape: [3, 1]
    result = a + b  # Broadcasting works now
    ```
    
2. **Preparing Input for Models**: Models often expect inputs with specific shapes, like `[batch_size, channels, height, width]`.
    
    ```python
    img = torch.tensor([1, 2, 3])  # Shape: [3]
    img = img.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3]
    ```
    
3. **Aligning Dimensions for Matrix Multiplication**: Certain operations, like `matmul`, require specific dimensional alignment.
