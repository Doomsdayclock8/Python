### **What is a Jacobian?**

The **Jacobian matrix** is a matrix of all first-order partial derivatives of a vector-valued function. If we have a function f:Rn→Rmf: \mathbb{R}^n \to \mathbb{R}^mf:Rn→Rm, where:

- x∈Rnx \in \mathbb{R}^nx∈Rn is the input (a vector with nnn elements),
- f(x)∈Rmf(x) \in \mathbb{R}^mf(x)∈Rm is the output (a vector with mmm elements),

the Jacobian J is an m×nm \times nm×n matrix defined as:

​​​​The Jacobian matrix of a vector-valued function $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^m$ is defined as:

$$
J(\mathbf{F}) =
\begin{bmatrix}
\frac{\partial F_1}{\partial x_1} & \frac{\partial F_1}{\partial x_2} & \cdots & \frac{\partial F_1}{\partial x_n} \\
\frac{\partial F_2}{\partial x_1} & \frac{\partial F_2}{\partial x_2} & \cdots & \frac{\partial F_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial F_m}{\partial x_1} & \frac{\partial F_m}{\partial x_2} & \cdots & \frac{\partial F_m}{\partial x_n}
\end{bmatrix}.
$$


Each row of the Jacobian corresponds to the gradient of one output component $fi(x)f_i(x)fi​(x) with respect to all inputs.