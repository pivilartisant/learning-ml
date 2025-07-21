Here's your **NumPy Quick Reference** in Markdown format:

---

````markdown
# 🧠 NumPy Quick Reference

NumPy (`numpy`) is the core Python library for numerical computing.

## 📦 1. Creating Arrays

```python
import numpy as np

a = np.array([1, 2, 3])                    # 1D array
b = np.array([[1, 2], [3, 4]])             # 2D array
zeros = np.zeros((2, 3))                   # 2x3 zero matrix
ones = np.ones((2, 3))                     # 2x3 ones matrix
arange = np.arange(0, 10, 2)               # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)            # 5 points between 0 and 1
identity = np.eye(3)                       # 3x3 identity matrix
````

---

## 🔄 2. Transposing & Reshaping

```python
A = np.array([[1, 2], [3, 4]])
A.T                           # Transpose: [[1, 3], [2, 4]]

B = np.arange(6).reshape(2, 3)  # Reshape 1D → 2D
B.flatten()                     # Flatten to 1D
```

---

## 🧷 3. Indexing & Slicing

```python
arr = np.array([[10, 20, 30], [40, 50, 60]])

arr[0, 1]        # 20
arr[1, :]        # [40 50 60]
arr[:, 2]        # [30 60]
arr[0:2, 1:3]    # [[20 30], [50 60]]

# Fancy indexing
arr[[0, 1], [1, 2]]   # [20 60]

# Boolean indexing
arr[arr > 30]         # [40 50 60]
```

---

## ➕ 4. Mathematical Operations

```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

x + y                # [5 7 9]
x * y                # [4 10 18] (element-wise)
x ** 2               # [1 4 9]
np.exp(x)            # e^x
np.log(x)            # ln(x)
np.sqrt(x)           # square root

np.sum(x)
np.mean(x)
np.std(x)
np.max(x)
np.min(x)
```

**Broadcasting Example:**

```python
x = np.array([[1], [2], [3]])   # Shape (3,1)
y = np.array([10, 20, 30])      # Shape (3,)
x + y                           # Broadcast to (3,3)
```

---

## 📐 5. Linear Algebra

```python
from numpy.linalg import inv, det, eig, norm

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B           # Matrix multiplication
inv(A)          # Inverse of A
det(A)          # Determinant of A
eig(A)          # Eigenvalues & eigenvectors
norm(A)         # Frobenius norm
np.dot(A, B)    # Same as A @ B
```

---

## 🎲 6. Random Sampling

```python
np.random.seed(42)

np.random.rand(2, 3)               # Uniform [0, 1)
np.random.randn(2, 3)              # Normal distribution
np.random.randint(0, 10, (3, 3))   # Random ints [0, 10)

np.random.choice([1, 2, 3], size=5, replace=True)
np.random.permutation([1, 2, 3, 4])
```

---

## 🧹 Bonus: Array Utilities

```python
np.clip(x, 0, 5)             # Limit values to [0, 5]
np.unique([1, 2, 2, 3])      # [1 2 3]
np.isnan(np.array([1, np.nan, 2]))   # [False True False]
np.argsort(x)               # Sort indices
```

---

```

Would you like me to save this as a `.md` file and send it to you for download?
```