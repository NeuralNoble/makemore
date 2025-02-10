

# **Understanding How PyTorch Stores Tensors in Memory**

## **1. PyTorch Tensors are Stored as 1D Arrays**
- In PyTorch, all tensors are stored as a **1D sequence of numbers** in memory, regardless of their actual shape.
- The multi-dimensional structure we see (e.g., 2D, 3D tensors) is just a **view** controlled by **shape** and **stride**.

---

## **2. Shape and Stride - The Key to Interpretation**
- **Shape**: Defines the number of elements along each dimension.
- **Stride**: Defines how many steps to move in memory to go from one element to the next along each dimension.

For example:
```python
import torch

t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6]])

print("Tensor:\n", t)
print("Storage:", t.storage())  
print("Shape:", t.shape)  
print("Stride:", t.stride())  
```
### **Output:**
```
Tensor:
 tensor([[1, 2, 3],
         [4, 5, 6]])
Storage:  1, 2, 3, 4, 5, 6
Shape: torch.Size([2, 3])
Stride: (3, 1)
```
- **Storage**: `[1, 2, 3, 4, 5, 6]` (stored as a 1D array)
- **Shape**: `(2, 3)` means we view it as 2 rows and 3 columns.
- **Stride**: `(3,1)`
  - Moving **one row down** → Jump **3 steps** in memory.
  - Moving **one column right** → Jump **1 step** in memory.

---

## **3. Understanding Stride with an Index Table**
| Index | Value | Logical Position (Row, Col) | Memory Offset |
|--------|------|---------------------------|--------------|
| 0      | 1    | (0,0)                      | `0*3 + 0*1 = 0` |
| 1      | 2    | (0,1)                      | `0*3 + 1*1 = 1` |
| 2      | 3    | (0,2)                      | `0*3 + 2*1 = 2` |
| 3      | 4    | (1,0)                      | `1*3 + 0*1 = 3` |
| 4      | 5    | (1,1)                      | `1*3 + 1*1 = 4` |
| 5      | 6    | (1,2)                      | `1*3 + 2*1 = 5` |

Each element is stored in a **continuous block of memory**, and PyTorch calculates its position using **stride values**.

---

## **4. How Reshaping Works**
Since the data is stored **linearly**, PyTorch can **reshape** a tensor without copying memory:

```python
t_reshaped = t.view(3, 2)  # Change to 3 rows, 2 columns
print(t_reshaped)
```
### **Output:**
```
tensor([[1, 2],
        [3, 4],
        [5, 6]])
```
- The **underlying storage is still `[1, 2, 3, 4, 5, 6]`**.
- **New stride**: `(2, 1)`, meaning:
  - Moving **one row down** → Jump **2 steps**.
  - Moving **one column right** → Jump **1 step**.

---

## **5. What Happens When We Transpose?**
Transposing swaps dimensions but **does not change the underlying storage**.

```python
t_T = t.T  # Transpose
print("Transposed Tensor:\n", t_T)
print("Storage:", t_T.storage())  
print("Stride:", t_T.stride())  
```
### **Output:**
```
Transposed Tensor:
 tensor([[1, 4],
         [2, 5],
         [3, 6]])
Storage: 1, 2, 3, 4, 5, 6
Stride: (1, 3)
```
- **Storage remains `[1, 2, 3, 4, 5, 6]`**.
- **Stride changes to `(1, 3)`**:
  - Now, to move **one row down**, jump **1 step**.
  - To move **one column right**, jump **3 steps**.

🚨 **Problem:** Many operations (like `.sum()`) can be **slow** on non-contiguous tensors because of inefficient memory access.

---

## **6. When PyTorch Copies Memory**
Some operations **don’t just change metadata but actually allocate new memory**, such as:
- `.clone()`
- `.contiguous()`
- `.reshape()` (if it needs a different layout)
- `.permute()` (when followed by tensor operations)

### **Fixing Slow Computation with `.contiguous()`**
If a tensor has a non-contiguous stride (like after `.T`), use `.contiguous()` to copy the data into **continuous memory**:
```python
t_T_cont = t_T.contiguous()  # Creates a new memory copy
```
- This **uses extra memory** but makes operations faster.

---

## **7. Summary**
✅ **PyTorch tensors are stored as 1D arrays in memory.**  
✅ **Shape and stride control how we "see" the tensor.**  
✅ **Reshaping, transposing, and permuting don’t copy data—they just change metadata.**  
✅ **Operations on non-contiguous tensors can be slow. Use `.contiguous()` when needed.**  

🚀 **PyTorch optimizes memory use, but understanding stride helps you write efficient code!**  

---


In PyTorch, `F.cross_entropy()` is a combination of **log softmax** and **negative log-likelihood loss (NLLLoss)**. Mathematically, it is given by:


$$
\text{CrossEntropyLoss}(\mathbf{x}, y) = - \log \left( \frac{e^{x_{y}}}{\sum_{j} e^{x_j}} \right)
$$

where:
- $\mathbf{x} = (x_1, x_2, ..., x_C)$ is the raw **logits** (unnormalized scores) from the model for $C$ classes.
- $y$ is the ground truth class index $(y \in \{0, 1, ..., C-1\})$.
- $x_y$ is the logit corresponding to the correct class.
- $\sum_{j} e^{x_j}$ is the sum of exponentiated logits across all classes.

### For a batch of size \(N\):
$$
\text{CrossEntropyLoss}(\mathbf{X}, \mathbf{Y}) = \frac{1}{N} \sum_{i=1}^{N} - \log \left( \frac{e^{x_{i,y_i}}}{\sum_{j} e^{x_{i,j}}} \right)
$$


where:
- $\mathbf{X}$ is the matrix of logits (shape: $N \times C$.
- $\mathbf{Y}$ is the vector of ground truth class indices.

**Logits are the raw, unnormalized scores output by a neural network before applying a probability function like softmax or sigmoid. They can take any real value, positive or negative.**

### Notes:
- `F.cross_entropy()` **expects logits, not probabilities**, because it internally applies `log_softmax()` before computing NLLLoss.
  - If **class weights** \(w_y\) are provided, the loss becomes:
  
  $\text{CrossEntropyLoss}(\mathbf{X}, \mathbf{Y}) = \frac{1}{N} \sum_{i=1}^{N} -w_{y_i} \log \left( \frac{e^{x_{i,y_i}}}{\sum_{j} e^{x_{i,j}}} \right)$
  
  