To import `.py` files from a folder with spaces in the path one can use one of the following methods in Python:

---

### Method 1: **Add the Folder to `sys.path`**

Use `sys.path` to add the folder containing the `.py` files to Python's import path.

```python
import sys
import os

# Define the path to your folder
folder_path = r"C:\Users\Tawfique\Thesis\Data_Augmentation\Other Models\AutoDiffusion"

# Add the folder to sys.path
sys.path.append(folder_path)

# Now you can import the Python files from this folder
import my_script  # Replace 'my_script' with the name of your .py file (without .py extension)
```
### Note: use raw string here ([[Use of raw string in handling path]])
**Example:**  
If the folder contains a file `helper.py`:

```python
# Import the file
import helper

# Use its functions
helper.some_function()
```

---

### Method 2: **Using `importlib` to Dynamically Import**

If you know the exact file path, you can use `importlib` to load the file dynamically:

```python
import importlib.util
import os

# Path to your Python file
file_path = r"C:\Users\Tawfique\Thesis\Data_Augmentation\Other Models\AutoDiffusion\helper.py"

# Dynamically load the module
module_name = os.path.basename(file_path).split('.')[0]  # Get module name (e.g., 'helper')
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Use functions from the imported module
module.some_function()
```

---

### Method 3: **Modify `PYTHONPATH` Environment Variable**

1. Add the folder path to the `PYTHONPATH` environment variable:
    
    - Open System Properties > Environment Variables.
    - Add `C:\Users\Tawfique\Thesis\Data_Augmentation\Other Models\AutoDiffusion` to the `PYTHONPATH`.
2. Restart your Python interpreter and directly import:
    
    ```python
    import helper  # Replace 'helper' with your .py file name
    ```
    

---

### Recommendations:

- Use **Method 1** if the folder path changes dynamically in your script.
- Use **Method 3** for a more permanent solution if you often import files from this location.

Let me know which method works for you!