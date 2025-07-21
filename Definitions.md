# **Model**:
In machine learning, a model is a mathematical function or computational structure that maps **input data** to **predictions or outputs**. It is defined by a set of **parameters** (like weights and biases) that are learned from training data.

# **Tensors**: 
are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

# **Neural Network**: 
A computational model composed of layers of interconnected nodes (called **neurons**), inspired by the structure of the human brain. Each neuron applies a mathematical transformation to its inputs and passes the result to the next layer. Neural networks learn by adjusting their internal parameters (weights and biases) to minimize a loss function based on training data.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network's parameters
- Update the weights of the network, typically using a simple update rule: `weight = weight - learning_rate * gradient`

# **Learning Signal**
refers to the information used by a machine learning model to update its parameters in order to improve performance. In neural networks, this signal is typically the **gradient of the loss function with respect to the model parameters**, computed during backpropagation. It tells the model how to change each weight to reduce prediction error on future examples.

# **Loss**: 
A scalar value that quantifies the difference between the model’s predicted output and the true target. It is computed using a **loss function** (e.g., Mean Squared Error, Cross-Entropy) and represents how well or poorly the model is performing on a given input.

# **Gradient**: 
The partial derivative of the loss function with respect to each model parameter (e.g., weights and biases). It indicates the direction and magnitude of change needed for each parameter to minimize the loss. Gradients are computed via **backpropagation** and used by optimization algorithms (e.g., SGD, Adam) to update parameters.

# **Forward Propagation**:
In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.

# **Backward Propagation**: 
In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (_gradients_), and optimizing the parameters using gradient descent. For a more detailed walkthrough of backprop, check out this [video from 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8).

# **Automatic Differentiation (autodiff)**: 
A technique used to compute exact derivatives of functions expressed as computer programs. Unlike symbolic differentiation or numerical approximation, autodiff breaks computations into elementary operations (like addition, multiplication) and applies the **chain rule** to compute gradients efficiently and accurately.

# **Derivative**: 
The derivative of a function measures how the function’s output changes as its input changes. In simple terms, it tells you the **slope** or **rate of change** of a function at a given point. In machine learning, derivatives are used to understand **how much the loss changes** when a parameter (like a weight in a neural network) changes — this is crucial for optimization.

# **Convolution**:
In deep learning, convolution is a mathematical operation used to extract features from data (usually images) by sliding a small filter (also called a **kernel**) over the input. At each position, the filter computes a **dot product** between its values and the overlapping region of the input.

This operation helps detect **patterns** like edges, textures, or shapes, and is a core building block of **Convolutional Neural Networks (CNNs)**.

# **Convolutional Neural Networks (CNNs)**: 
A type of deep learning model designed to process data with a grid-like structure, such as images. CNNs use **convolutional layers** to automatically and adaptively learn spatial features like edges, textures, and shapes by applying filters (kernels) across the input.

CNNs are particularly effective for tasks like **image classification**, **object detection**, and **image segmentation**.