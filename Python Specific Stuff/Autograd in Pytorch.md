# 🧠 Understanding Autograd in PyTorch

## What is Autograd?

Autograd is PyTorch’s tool for **automatic differentiation**.  
It tracks operations on tensors that have `requires_grad=True`, so it can automatically compute **gradients** during the backward pass.

---

## 1. Creating Tensors That Track History

When you create tensors with `requires_grad=True`, you're telling PyTorch:  
*"I want to track how changes to this tensor affect future computations."*

```python
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
````

---

## 2. Computation Builds a Graph

Every operation you do creates a **computation graph**.  
This graph remembers how the final output depends on each input.

```python
Q = 3 * a**3 - b**2
```

PyTorch now knows how `Q` depends on `a` and `b`.

---

## 3. The Backward Pass (Backpropagation)

Calling `.backward()` on `Q` tells PyTorch:  
👉 “Calculate how much each input (a, b) contributed to the output (Q).”

It computes the gradients using the **chain rule**, moving backwards through the graph.

---

## 4. Storing the Gradients

After `.backward()`, the gradients are stored in the `.grad` attribute of each tensor:

```python
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(a.grad)  # ∂Q/∂a = 9 * a**2
print(b.grad)  # ∂Q/∂b = -2 * b
```

---

## 5. Why Do We Pass a `gradient` Argument?

If `Q` is a vector (not a single number), PyTorch needs to know **how to combine the gradients**.  
You pass a tensor (`gradient=...`) that tells it how to weight each element of `Q` when calculating the final gradients.

Or you can just do:

```python
Q.sum().backward()
```

To turn it into a scalar before calling `.backward()`.

---

## 🧾 Summary

- Autograd tracks operations on tensors with `requires_grad=True`.
    
- It builds a computation graph of those operations.
    
- Calling `.backward()` computes **gradients** for every parameter involved.
    
- These gradients are stored in `.grad`, and used to update model parameters during training.
    

Autograd powers the learning process in PyTorch!