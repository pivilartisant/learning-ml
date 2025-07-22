# Understanding the RNN Update Rule:
`h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)`

This equation is the **core of a vanilla Recurrent Neural Network (RNN)**. It defines how the hidden state (memory) of the network updates at each time step.

---

## What does this equation represent?

The equation computes the **hidden state** \( h_t \) at time step \( t \). It combines:

- The previous hidden state \( h_{t-1} \)
- The current input \( x_t \)
- Two learnable weight matrices \( W_{hh} \) and \( W_{xh} \)

The result is passed through the non-linear activation function \( \tanh \) to produce the new hidden state.

---

## 🧩 Term-by-term breakdown

| Term          | Description                                               |
| ------------- | --------------------------------------------------------- |
| \( h_t \)     | Current hidden state (memory of the network)              |
| \( h_{t-1} \) | Hidden state from the previous time step                  |
| \( x_t \)     | Input at time step \( t \)                                |
| \( W_{hh} \)  | Weight matrix for hidden-to-hidden transitions            |
| \( W_{xh} \)  | Weight matrix for input-to-hidden projection              |
| ( tanh() )    | Activation function, squashes output to range \([-1, 1]\) |

---

## What’s happening?

- The network **combines past memory** (\( h_{t-1} \)) **with current input** (\( x_t \)).
- Applies a **linear transformation** using the weight matrices.
- Applies a **non-linear activation** via `tanh`.
- Produces a new hidden state \( h_t \), which carries the updated memory.

---

##  Visual Flow
```
Input (x_t) Previous Memory (h_{t-1})  
↓ ↓  
[ Weighted sum: W_xh * x_t + W_hh * h_{t-1} ]  
↓  
tanh (nonlinear activation)  
↓
New Memory (h_t)
```


## Why use `tanh`?

- It is zero-centered, which helps training converge faster.
- Keeps values bounded between -1 and 1.
- Helps with **vanishing gradients** (somewhat).

---

##  Summary

This equation allows an RNN to:

- Handle **sequential data** (text, audio, time series, etc.)
- **"Remember" previous context** while processing current inputs
- Learn temporal dependencies via its hidden state updates

> 💡 This is the fundamental building block behind more advanced models like LSTMs and GRUs.

---
