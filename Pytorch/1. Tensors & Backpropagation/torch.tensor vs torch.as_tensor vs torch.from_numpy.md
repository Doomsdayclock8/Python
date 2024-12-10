### **1. `torch.tensor`**

- **Purpose**: Creates a new tensor by **copying** data from the input.
- **Memory Behavior**: Always copies data; no memory sharing with the input.
- **Use Case**: When you need an independent PyTorch tensor and don’t want changes in the input to affect the tensor.
- **Supported Inputs**: Lists, tuples, NumPy arrays, tensors, scalars.

```python
import numpy as np
import torch

arr = np.array([1, 2, 3])
tensor = torch.tensor(arr)  # Copies data
tensor[0] = 10
print(arr)  # Output: [1 2 3] (unchanged)
```

---

### **2. `torch.as_tensor`**

- **Purpose**: Converts an input into a tensor. Tries to **reuse the input memory** if possible.
- **Memory Behavior**:
    - **Zero-copy**: If the input is a NumPy array or a PyTorch tensor (uses shared memory).
    - **Copies**: If the input is a Python list, tuple, or other types.
- **Use Case**: When you want to reuse existing memory whenever possible for efficiency.
- **Supported Inputs**: Lists, tuples, NumPy arrays, tensors, scalars.

```python
arr = np.array([1, 2, 3])
tensor = torch.as_tensor(arr)  # Zero-copy
tensor[0] = 10
print(arr)  # Output: [10  2  3] (shared memory)

lst = [1, 2, 3]
tensor = torch.as_tensor(lst)  # Copies data
tensor[0] = 10
print(lst)  # Output: [1, 2, 3] (unchanged)
```

---

### **3. `torch.from_numpy`**

- **Purpose**: Converts a NumPy array into a PyTorch tensor **without copying**.
- **Memory Behavior**: Always **zero-copy**; the tensor and the NumPy array share the same memory.
- **Use Case**: When working specifically with NumPy arrays and want efficient conversion without duplication.
- **Supported Inputs**: Only NumPy arrays.

```python
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)  # Zero-copy
tensor[0] = 10
print(arr)  # Output: [10  2  3] (shared memory)
```

---

### **Key Differences**

|Feature|`torch.tensor`|`torch.as_tensor`|`torch.from_numpy`|
|---|---|---|---|
|**Data Copy**|Always copies|Copies (for lists, etc.)|No copy (always zero-copy)|
|**Memory Sharing**|No|Only for tensors/arrays|Always|
|**Input Type Support**|Lists, arrays, tuples|Lists, arrays, tuples|Only NumPy arrays|
|**Speed**|Slower (due to copy)|Faster if no copy needed|Fastest (no copy)|

---

### **When to Use**

1. **`torch.tensor`**: When you need a completely independent tensor.
2. **`torch.as_tensor`**: When you prefer memory sharing for arrays or tensors but can handle copying for other types.
3. **`torch.from_numpy`**: When working with NumPy arrays and want efficient, zero-copy conversion.