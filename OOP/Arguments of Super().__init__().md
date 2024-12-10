### **`super().__init__()` in Python**

The `super().__init__()` function is used in Python to call the initializer of a parent class from within the initializer (`__init__`) of a child class. This is particularly important in class inheritance.

---

### **Key Points:**

1. **No Arguments for `super()` in Python 3**
    
    - In Python 3, `super()` no longer requires explicit arguments. It automatically resolves to the parent class of the current class.
    
    ```python
    class Parent:
        def __init__(self):
            print("Parent init")
    
    class Child(Parent):
        def __init__(self):
            super().__init__()
            print("Child init")
    
    obj = Child()  
    # Output:
    # Parent init
    # Child init
    ```
    

---

2. **Child `__init__` Must Match Parent’s `__init__` Arguments**
    
    - If the parent’s `__init__` accepts arguments, the child’s `__init__` must pass these arguments to `super().__init__()`.
    
    ```python
    class Parent:
        def __init__(self, name):
            self.name = name
            print(f"Parent init with {name}")
    
    class Child(Parent):
        def __init__(self, name, age):
            super().__init__(name)  # Pass `name` to Parent
            self.age = age
            print(f"Child init with {name}, age {age}")
    
    obj = Child("Alice", 25)
    # Output:
    # Parent init with Alice
    # Child init with Alice, age 25
    ```
    

---

3. **If Parent’s `__init__` is Empty/Doesn’t Exist/Default Behavior (`nn.Module`)**
    
    - If the parent class:
        - Has no `__init__`, or
        - Doesn’t require arguments (e.g., `nn.Module` in PyTorch),  
            the child’s `__init__` can skip calling or handle its arguments independently.
    
    ```python
    import torch.nn as nn
    
    class Parent(nn.Module):  # No __init__ explicitly defined
        pass
    
    class Child(Parent):
        def __init__(self):
            super().__init__()  # Works without any arguments
            print("Child init")
    
    obj = Child()
    # Output:
    # Child init
    ```
    

---

4. **Mismatched Arguments Cause `TypeError`**
    
    - If the child sends arguments that the parent’s `__init__` does not expect **OR** the parent expects arguments but the child does not send them, a `TypeError` occurs.
    
    ```python
    # Example of a mismatch
    class Parent:
        def __init__(self, name):
            print(f"Parent init with {name}")
    
    class Child(Parent):
        def __init__(self):
            super().__init__()  # Missing required `name` argument
            print("Child init")
    
    obj = Child()
    # Output: TypeError: __init__() missing 1 required positional argument: 'name'
    ```
    

---

### **Summary:**

- `super()` is used to access the parent class’s methods, especially `__init__`.
- If the parent’s `__init__` takes arguments, the child must provide them correctly.
- If the parent’s `__init__` is empty or doesn’t exist, the child’s `__init__` doesn’t need to provide anything.
- Mismatched arguments between child and parent result in a `TypeError`.