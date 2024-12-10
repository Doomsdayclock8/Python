To see the shape of a tensor in PyTorch, you can use the **`.shape` attribute** or the **`torch.Tensor.size()` method**.

---

### **Methods to Check the Shape of a Tensor**

#### 1. **Using `.shape` Attribute**

- Returns the shape of the tensor as a `torch.Size` object.
- This is the most concise and commonly used method.

```python
import torch

# Create a tensor
tensor = torch.randn(3, 4, 5)

# Check shape
print(tensor.shape)  # Output: torch.Size([3, 4, 5])
```

---

#### 2. **Using `.size()` Method**

- Functionally the same as `.shape`, but in method form.
- Useful if you want to query a specific dimension.

```python
# Check shape
print(tensor.size())  # Output: torch.Size([3, 4, 5])

# Query a specific dimension
print(tensor.size(1))  # Output: 4 (size of the second dimension)
```

---

### **Differences Between `.shape` and `.size()`**

- **`.shape`**: Attribute, directly accessed.
- **`.size()`**: Method, can take an argument to query a specific dimension.

---

### **Practical Examples**

#### 1. Check Total Number of Dimensions

```python
print(tensor.ndimension())  # Output: 3 (number of dimensions)
```

#### 2. Use Shape in Loops or Computations

```python
batch_size, num_features = tensor.shape[0], tensor.shape[1]
print(batch_size, num_features)  # Example: 3, 4
```

#### 3. Handle Dynamic Shapes

```python
tensor = torch.randn(8, 2, 10)  # Shape: [8, 2, 10]
print(f"Last dimension size: {tensor.shape[-1]}")  # Output: 10
```

---

### **Key Takeaways**

- Use `.shape` for simplicity and readability.
- Use `.size()` if you need to query specific dimensions.
- Both return the tensor's shape as a `torch.Size` object, which behaves like a tuple.****