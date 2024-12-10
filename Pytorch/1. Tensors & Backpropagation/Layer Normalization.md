Layer normalization is a technique used in deep learning to stabilize and accelerate the training of neural networks. Here's a concise explanation:

1. **Purpose**: It normalizes the activations (outputs) of a layer across all neurons in the layer, ensuring that their mean is 0 and variance is 1. This helps the model learn more effectively by reducing internal covariate shift (variations in input distributions across layers during training).
    
2. **How It Works**:
    
    - Given an input vector x\mathbf{x} of activations for a layer, calculate the mean μ\mu and variance σ2\sigma^2 for all elements in the vector.
    - Normalize each activation xix_i using: x^i=xi−μσ2+ϵ\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} where ϵ\epsilon is a small constant to prevent division by zero.
    - Scale and shift the normalized values using learned parameters γ\gamma (scale) and β\beta (shift): yi=γx^i+βy_i = \gamma \hat{x}_i + \beta
    - These learned parameters allow the model to undo normalization when it's beneficial.
3. **Key Difference from Batch Normalization**:
    
    - Layer normalization computes statistics per sample, across all features of the input, making it more suitable for tasks with small batch sizes or sequential data like RNNs.
4. **Intuition**: Imagine you're a runner training on various terrains (inputs). Layer normalization adjusts the ground so you're always running on flat terrain (mean = 0, variance = 1), ensuring consistent performance regardless of the initial conditions. The learned γ\gamma and β\beta allow the model to adjust the "ground" for optimal performance.
![[Pasted image 20241203023815.png]]