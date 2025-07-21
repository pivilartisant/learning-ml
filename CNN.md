[MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)
[LeNet](https://en.wikipedia.org/wiki/LeNet)

https://www.youtube.com/watch?v=2xqkSUhmmXU


# Defining a CNN
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # First convolutional layer:
        # - Takes 1 input channel (grayscale image)
        # - Produces 6 output channels (feature maps)
        # - Uses a 5x5 kernel/filter to scan over the image
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        # Second convolutional layer:
        # - Takes 6 input channels (from previous layer)
        # - Produces 16 output channels
        # - Uses a 5x5 kernel/filter
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Fully connected (linear) layers:
        # After the convolutions and pooling, the image is reduced in size.
        # The next layers act like a classic neural network, processing the "flattened" features.
        
        # The size 16*5*5 comes from the size of the feature maps after conv and pooling
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)  # Output 10 classes (e.g. digits 0-9)

    def forward(self, x):
        # Pass input through first conv layer, then apply ReLU activation function
        x = F.relu(self.conv1(x))  # Shape: (N, 6, 24, 24) if input is (N,1,28,28)
        
        # Apply max pooling to reduce spatial size by 2
        x = F.max_pool2d(x, kernel_size=2)  # Shape: (N, 6, 12, 12)
        
        # Second conv layer + ReLU
        x = F.relu(self.conv2(x))  # Shape: (N, 16, 8, 8)
        
        # Another max pooling to reduce size
        x = F.max_pool2d(x, kernel_size=2)  # Shape: (N, 16, 4, 4)
        
        # Flatten the tensor to feed into fully connected layers
        # Shape becomes (N, 16*4*4) = (N, 256)
        x = torch.flatten(x, start_dim=1)
        
        # Pass through fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))  # Shape: (N, 120)
        x = F.relu(self.fc2(x))  # Shape: (N, 84)
        
        # Final output layer (no activation here, usually combined with loss function like CrossEntropyLoss)
        x = self.fc3(x)  # Shape: (N, 10)
        
        return x


# Create the network instance
net = Net()

# Print the model structure
print(net)

```

# LeNet

**LeNet CNN** is one of the earliest and most famous **Convolutional Neural Networks** designed to recognize images, especially handwritten digits.

It uses a combination of **convolutional layers**, **pooling layers**, and **fully connected layers** to automatically learn and classify important features in images.

- **Convolutional layers**: Extract features like edges or shapes by sliding small filters (**kernels**) over the input image.
    
- **Pooling layers**: Reduce the spatial size of feature maps to focus on important features and reduce computation.
    
- **Fully connected layers**: Use the extracted features to classify the input into different categories (e.g., digits 0-9).
    
- **Activation functions** (like ReLU or sigmoid): Add non-linearity so the network can learn complex patterns.
    

#### Use case:

LeNet was originally developed for handwritten digit recognition (MNIST dataset) and paved the way for modern CNN architectures.

#### Typical architecture overview:

1. Input: grayscale image (28×28 pixels)
    
2. Convolution + activation (e.g., sigmoid or ReLU)
    
3. Pooling (subsampling)
    
4. Repeat convolution + pooling
    
5. Flatten features
    
6. Fully connected layers
    
7. Output layer with scores for each class


# Beginner friendly CNN

```python
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Step 1: Load and transform the dataset
# We will use the MNIST dataset of handwritten digits.
# We transform the images to tensors and normalize pixel values to [0,1].
transform = transforms.ToTensor()

# Download and load the training dataset
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
# Download and load the test dataset
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Create data loaders to iterate over the datasets in batches
# batch_size defines how many samples per batch to load
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Step 2: Define the neural network architecture by subclassing nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers:
        # Flatten layer converts 2D images (28x28) into 1D vectors of size 784
        self.flatten = nn.Flatten()
        # Fully connected linear layer: 784 inputs -> 512 outputs (neurons)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # First linear layer
            nn.ReLU(),              # ReLU activation function adds non-linearity
            nn.Linear(512, 512),    # Second linear layer
            nn.ReLU(),
            nn.Linear(512, 10)      # Output layer: 10 classes (digits 0-9)
        )

    def forward(self, x):
        # Define forward pass (how data flows through the network)
        x = self.flatten(x)              # Flatten input images
        logits = self.linear_relu_stack(x)  # Pass through layers
        return logits                   # Return raw scores (logits)

# Step 3: Instantiate the network and define loss function and optimizer
model = NeuralNetwork()

# Loss function: CrossEntropyLoss combines softmax and negative log likelihood
# It compares predicted class scores to true labels
loss_fn = nn.CrossEntropyLoss()

# Optimizer: Stochastic Gradient Descent (SGD)
# Updates model parameters based on gradients to minimize loss
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Step 4: Define training loop for one epoch
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # Number of samples in training set
    model.train()                   # Set model to training mode
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)             # Forward pass
        loss = loss_fn(pred, y)     # Compute loss

        # Backpropagation
        optimizer.zero_grad()       # Clear gradients from previous step
        loss.backward()             # Compute gradients via backpropagation
        optimizer.step()            # Update model parameters

        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

# Step 5: Define testing loop to evaluate model accuracy on test data
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)   # Number of samples in test set
    num_batches = len(dataloader)    # Number of batches
    model.eval()                     # Set model to evaluation mode
    test_loss, correct = 0, 0

    with torch.no_grad():            # Disable gradient calculations for evaluation
        for X, y in dataloader:
            pred = model(X)          # Forward pass
            test_loss += loss_fn(pred, y).item()  # Sum up batch loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Count correct predictions

    test_loss /= num_batches
    accuracy = correct / size

    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Step 6: Run the training and testing process for multiple epochs
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Training complete!")

```