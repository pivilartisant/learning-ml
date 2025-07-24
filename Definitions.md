# **Model**:
In machine learning, a model is a mathematical function or computational structure that maps **input data** to **predictions or outputs**. It is defined by a set of **parameters** (like weights and biases) that are learned from training data.

----

# Intuition in Machine Learning

**Intuition** in machine learning refers to a **natural understanding or insight** about how algorithms work and why certain methods perform better than others. It is the *mental grasp* or *feel* for the underlying patterns, behaviors, and principles of models and data without relying solely on formal proofs or complex mathematics.

In practice, intuition helps practitioners:
- Understand how model parameters influence predictions
- Anticipate how changes to data or algorithms affect results
- Diagnose problems such as overfitting, underfitting, or bias
- Choose the right model or approach for a given task

**Example:** Intuition about how increasing the number of training examples generally improves a model’s performance, or that simpler models might generalize better than overly complex ones.

---

# **Tensors**: 
are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

----

# **Neural Network**: 
A computational model composed of layers of interconnected nodes (called **neurons**), inspired by the structure of the human brain. Each neuron applies a mathematical transformation to its inputs and passes the result to the next layer. Neural networks learn by adjusting their internal parameters (weights and biases) to minimize a loss function based on training data.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network's parameters
- Update the weights of the network, typically using a simple update rule: `weight = weight - learning_rate * gradient`

----


# **Learning Signal**
refers to the information used by a machine learning model to update its parameters in order to improve performance. In neural networks, this signal is typically the **gradient of the loss function with respect to the model parameters**, computed during backpropagation. It tells the model how to change each weight to reduce prediction error on future examples.

----

# **Loss**: 
A scalar value that quantifies the difference between the model’s predicted output and the true target. It is computed using a **loss function** (e.g., Mean Squared Error, Cross-Entropy) and represents how well or poorly the model is performing on a given input.

----
# **Gradient**: 
The partial derivative of the loss function with respect to each model parameter (e.g., weights and biases). It indicates the direction and magnitude of change needed for each parameter to minimize the loss. Gradients are computed via **backpropagation** and used by optimization algorithms (e.g., SGD, Adam) to update parameters.

---
# **Learning Rate** 
is a **hyperparameter** that controls how much a model’s weights are adjusted during training in response to the estimated error (loss).

[See Learning Rate Scheduler](https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863/)


----
# **Forward Propagation**:
In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.

----
# **Backward Propagation**: 
In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (_gradients_), and optimizing the parameters using gradient descent. For a more detailed walkthrough of backprop, check out this [video from 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8).

----
# **Automatic Differentiation (autodiff)**: 
A technique used to compute exact derivatives of functions expressed as computer programs. Unlike symbolic differentiation or numerical approximation, autodiff breaks computations into elementary operations (like addition, multiplication) and applies the **chain rule** to compute gradients efficiently and accurately.

----
# **Derivative**: 
The derivative of a function measures how the function’s output changes as its input changes. In simple terms, it tells you the **slope** or **rate of change** of a function at a given point. In machine learning, derivatives are used to understand **how much the loss changes** when a parameter (like a weight in a neural network) changes — this is crucial for optimization.

----
# **Convolution**:
In deep learning, convolution is a mathematical operation used to extract features from data (usually images) by sliding a small filter (also called a **kernel**) over the input. At each position, the filter computes a **dot product** between its values and the overlapping region of the input.

This operation helps detect **patterns** like edges, textures, or shapes, and is a core building block of **Convolutional Neural Networks (CNNs)**.

----
# **Convolutional Neural Networks (CNNs)**: 
A type of deep learning model designed to process data with a grid-like structure, such as images. CNNs use **convolutional layers** to automatically and adaptively learn spatial features like edges, textures, and shapes by applying filters (kernels) across the input.

CNNs are particularly effective for tasks like **image classification**, **object detection**, and **image segmentation**.

----

#  **Softmax classifier**
A **Softmax classifier** is a type of **multiclass classification model** commonly used in machine learning and deep learning, especially as the final layer of a neural network when the task involves categorizing input into one of several classes.
#### **How It Works**

- The softmax function **amplifies differences** between scores.
- Higher input values get mapped to **higher probabilities**, but all outputs sum to 1, making them interpretable as **class probabilities**.
- The model **predicts the class** with the **highest softmax score**.

In training, the softmax classifier is typically used in combination with **cross-entropy loss**, which measures the difference between the predicted probability distribution and the true distribution (one-hot encoded labels).

----

# Cross-Entropy Loss

**Cross-Entropy Loss** is a commonly used loss function in classification tasks, especially when outputs represent probabilities (e.g., via a softmax function). It measures the difference between two probability distributions: the true labels and the predicted probabilities.

####  **Intuition**

- If the model predicts a high probability for the correct class → **low loss**.
- If it predicts a low probability for the correct class → **high loss**.
- Cross-entropy **penalizes incorrect confident predictions** heavily.


####  **Why It's Used**

- Encourages the model to assign **high probability to the correct class**.
- Differentiable → suitable for **gradient-based optimization**.
- Works well with softmax output layers in **multiclass classification**.

----

# Mini-batch Stochastic Gradient Descent (Mini-batch SGD)

**Definition:**  
Mini-batch Stochastic Gradient Descent is an optimization algorithm used to train machine learning models, especially neural networks. It combines ideas from both batch gradient descent and stochastic gradient descent by updating model parameters using a small, randomly selected subset ("mini-batch") of the training data at each iteration.



### Key points:

- Instead of using the **entire dataset** (as in batch gradient descent) or just a **single example** (as in stochastic gradient descent), mini-batch SGD uses a **small batch of examples** (typically between 32 and 512 samples).
- This approach balances **computational efficiency** and **stable convergence**.
- It enables faster computation by leveraging vectorized operations and parallelism on hardware like GPUs.
- Using mini-batches reduces the variance of the parameter updates compared to pure SGD, leading to more stable convergence.



### Basic workflow:

1. Split the training dataset into mini-batches.
2. For each mini-batch:
   - Compute the gradient of the loss with respect to model parameters using only the mini-batch.
   - Update parameters by moving in the direction opposite to the gradient.
3. Repeat for multiple epochs until convergence.


----

# Gradient Descent

**Definition:**  
Gradient Descent is an iterative optimization algorithm used to minimize a function, commonly a loss function in machine learning. It works by repeatedly moving the model parameters in the direction of the steepest descent (negative gradient) of the function to find the parameters that minimize the loss.

### Key points:

- Used to **find the minimum** of a differentiable function.
- At each iteration, it updates parameters by taking a step proportional to the **negative gradient** of the function at the current point.
- The size of the step is controlled by the **learning rate**.
- Widely used for training machine learning models, especially in neural networks.

### Intuition:

- The gradient points in the direction of the steepest **increase** of the function.
- Moving in the **opposite** direction reduces the function value.
- Repeating this process leads parameters closer to a local (or global) minimum


----

# Stochastic Gradient Descent (SGD)

**Definition:**  
Stochastic Gradient Descent is an optimization algorithm that updates model parameters using the gradient computed from a **single randomly selected training example** at each iteration, instead of the full dataset.

---

### Key points:

- Unlike batch gradient descent (which uses the entire dataset), SGD updates parameters **one sample at a time**.
- This leads to faster updates and can help the model escape local minima due to its noisy updates.
- However, the updates have higher variance, which can cause the loss to fluctuate during training.
- Often used when datasets are large, making full-batch computation expensive or impractical.