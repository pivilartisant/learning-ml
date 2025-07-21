```python
import torch

from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

data = torch.rand(1, 3, 64, 64) # 1 image, 3 channels, 64 width, 64 height 

labels = torch.rand(1, 1000) # one image for 1000 labels 

prediction = model(data) # forward pass


# We use the model's prediction and the corresponding label to calculate the error (`loss`). The next step is to backpropagate this error through the network. Backward propagation is kicked off when we call `.backward()` on the error tensor.

loss = (prediction - labels).sum()
loss.backward() # backward pass

# load an optimizer, in this case SGD with a learning rate of 0.01 and momentum of 0.9.
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# call `.step()` to initiate gradient descent. The optimizer adjusts each parameter by its gradient stored in `.grad`
optim.step() #gradient descent
```


