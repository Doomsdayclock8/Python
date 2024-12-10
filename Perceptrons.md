### Perceptrons in PyTorch

- A **perceptron** is the simplest type of artificial neural network, consisting of a single layer of neurons. 
- It is a binary classifier that decides whether an input belongs to one class or another.
- Geometrically its a n Dimension Plane where n is the no. of input features /Predictors
- Whatever dimension the tensor is in , it can only separate the data into 2 region, hence a binary classifier

### How it Works:

1. **Input**: Takes multiple inputs (features of data) with associated weights.
2. **Weighted Sum**: Computes the weighted sum of the inputs. $$z=∑i=1n(wi⋅xi)+bz = \sum_{i=1}^n (w_i \cdot x_i) + b$$ where $w_i$ are weights, $x_i$ are inputs, and b is the bias.
3. **Activation Function**: Passes the result through a step function: $$
   output={1if z≥00if z<0\text{output} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}
   $$
4. **Output**: Produces a binary result (1 or 0).

### Key Points:

- The perceptron **learns weights and bias** during training using the perceptron learning algorithm.
- It works only for **linearly separable data** and cannot handle more complex problems (like XOR).

For multi-class or non-linear problems, more advanced networks like multi-layer perceptrons (MLPs) are used.

To classify binary classes, the perceptron uses **Softmax** (or Sigmoid in some cases) as the activation function. **Softmax** converts raw scores into probabilities.

---

### Implementation in PyTorch

Below is a simple implementation of a perceptron using PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Example perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_dim, 2)  # Output size 2 for binary classification

    def forward(self, x):
        logits = self.linear(x)  # Linear combination
        probs = F.softmax(logits, dim=1)  # Softmax for probabilities
        return probs

# Create dummy data
input_dim = 3  # Example input features
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Two samples
y = torch.tensor([0, 1])  # Labels

# Initialize the perceptron
model = Perceptron(input_dim)

# Forward pass
outputs = model(X)
print("Predicted probabilities:\n", outputs)

# Output explanation:
# Each row contains two probabilities, one for each class (binary classification).
# The higher probability determines the predicted class.
```

---

### Key Points:

1. **Linear Equation**: Perceptrons are based on y=w⋅x+by = w \cdot x + b.
2. **Softmax Activation**: Converts logits into probabilities.
3. **Binary Classification**: Outputs probabilities for two classes; the higher one determines the prediction.
4. **nD Input Support**: Works with nn-dimensional inputs; you can scale it by increasing input dimensions.

---

### Example Output

For the dummy data in the example:

```plaintext
Predicted probabilities:
 tensor([[0.3543, 0.6457],
         [0.2562, 0.7438]])
```

This shows probabilities for two classes (columns). For the first sample, the predicted class is 1 (0.6457 > 0.3543).