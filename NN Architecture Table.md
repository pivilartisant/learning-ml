| Architecture      | Data Type          | Use Cases                      | Key Features                |
| ----------------- | ------------------ | ------------------------------ | --------------------------- |
| Feedforward (MLP) | Fixed-size vectors | Classification, regression     | Simple, fully connected     |
| CNN               | Images, grids      | Image/video tasks              | Convolutions, pooling       |
| RNN               | Sequences          | Sequential data tasks          | Memory via recurrence       |
| LSTM/GRU          | Sequences          | Long sequence dependencies     | Gated memory units          |
| Transformer       | Sequences          | NLP, translation               | Self-attention, parallelism |
| Autoencoder       | Various            | Compression, anomaly detection | Encoder-decoder structure   |
| GAN               | Various            | Generative tasks               | Adversarial training        |
| GNN               | Graphs             | Graph data tasks               | Graph convolutions          |




| Model Name                  | Architecture          | Description / Use Cases                                         |
|-----------------------------|-----------------------|----------------------------------------------------------------|
| LeNet-5                     | CNN                   | Early CNN for digit recognition (MNIST dataset).               |
| AlexNet                     | CNN                   | Breakthrough CNN for ImageNet classification (2012).           |
| VGGNet                      | CNN                   | Deep CNN with uniform 3x3 conv layers, popular for vision.     |
| ResNet                      | CNN with Residuals    | CNN with skip connections to enable very deep networks.        |
| Inception (GoogLeNet)       | CNN with Inception Modules | Uses multiple filter sizes in parallel, efficient and deep. |
| RNN                         | RNN                   | Basic recurrent network for sequential data.                   |
| LSTM                        | RNN variant           | Long Short-Term Memory network for sequences with long dependencies. |
| GRU                         | RNN variant           | Gated Recurrent Unit, simpler and faster than LSTM.            |
| Transformer                 | Attention-based       | State-of-the-art for NLP tasks, e.g., BERT, GPT.                |
| BERT                        | Transformer           | Pretrained Transformer for bidirectional language understanding. |
| GPT                         | Transformer           | Generative Transformer for text generation.                     |
| Autoencoder                 | Encoder-decoder       | For dimensionality reduction, anomaly detection, generative models. |
| Variational Autoencoder (VAE)| Autoencoder variant  | Probabilistic generative model.                                 |
| GAN                         | Adversarial model     | Generative model with generator and discriminator networks.    |
| Graph Convolutional Network (GCN) | GNN              | For graph-structured data.                                      |


| Concept                             | Description                                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Neuron / Node**                   | Basic unit of a neural network, simulates a biological neuron by applying a weighted sum and activation function. |
| **Layers**                          | Networks are composed of layers: Input, Hidden, and Output layers. Each layer contains multiple neurons.          |
| **Weights and Biases**              | Parameters learned during training that transform input data through the network.                                 |
| **Activation Functions**            | Functions that introduce non-linearity to the model (e.g., ReLU, Sigmoid, Tanh).                                  |
| **Loss Function**                   | A metric to measure how far the network’s prediction is from the true value (e.g., MSE, Cross-Entropy).           |
| **Optimizer**                       | Algorithm used to update weights to minimize the loss (e.g., SGD, Adam).                                          |
| **Forward Propagation**             | Process of passing input data through the network to generate output predictions.                                 |
| **Backward Propagation (Backprop)** | Algorithm to compute gradients of loss w.r.t. weights to update them during training.                             |
| **Epoch**                           | One full pass through the entire training dataset.                                                                |
| **Batch and Mini-batch**            | Splitting data into smaller groups for efficient training updates.                                                |
| **Overfitting**                     | When a model performs well on training data but poorly on unseen data.                                            |
| **Regularization**                  | Techniques to reduce overfitting, like dropout, L2 regularization.                                                |
| **Learning Rate**                   | Controls the step size during weight updates; crucial for training stability and speed.                           |