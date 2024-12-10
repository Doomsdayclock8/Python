In PyTorch, the method that is automatically invoked when you call a model object (e.g., `output = model(input)`) **must be named `forward`**. This is because PyTorch's `__call__` method in the `torch.nn.Module` class is hardcoded to call the `forward` method of your class.

### Why Does It Have to Be Named `forward`?

1. **Hardcoded in `__call__`**:
    
    - PyTorch’s `__call__` method is defined to look for a method named `forward` in your model class.
    - If `forward` is not defined, calling the model object will raise an error.
2. **Consistency**:
    
    - PyTorch uses `forward` as the standard name for defining the forward pass, ensuring consistency across models.

---

### What Happens If You Use Another Name?

If you use a name other than `forward`, the `__call__` method won’t find it, and you’ll get an error when calling the model object.

#### Example:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    # Custom name for forward pass
    def my_custom_forward(self, x):
        return self.linear(x)

# Instantiate the model
model = MyModel()

# Attempt to call the model
x = torch.randn(1, 10)
output = model(x)  # This will raise an error
```

**Error**:

```
AttributeError: 'MyModel' object has no attribute 'forward'
```

---

### How to Use a Custom Method Name?

If you absolutely need to use a custom name, you can override the `__call__` method in your model to call your custom method instead of `forward`:

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def my_custom_forward(self, x):
        return self.linear(x)

    # Override __call__ to use a custom method
    def __call__(self, *args, **kwargs):
        return self.my_custom_forward(*args, **kwargs)

# Instantiate and call the model
model = MyModel()
x = torch.randn(1, 10)
output = model(x)  # Now it works
```

---

### Best Practice

Stick with the `forward` method for defining the forward pass unless you have a very specific reason to do otherwise. Using `forward` ensures compatibility with PyTorch's ecosystem and avoids unnecessary complexity.