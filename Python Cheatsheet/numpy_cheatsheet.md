# Complete NumPy Cheat Sheet

## 1. Import and Basic Setup

```python
import numpy as np

# Check version
print(np.__version__)
```

## 2. Array Creation

### Basic Array Creation
```python
# From lists
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([[1, 2], [3, 4]])

# From tuples
arr3 = np.array((1, 2, 3))

# Specify data type
arr4 = np.array([1, 2, 3], dtype=float)
arr5 = np.array([1, 2, 3], dtype=np.int32)
```

### Special Arrays
```python
# Zeros
zeros = np.zeros(5)           # [0. 0. 0. 0. 0.]
zeros_2d = np.zeros((3, 4))   # 3x4 array of zeros

# Ones
ones = np.ones(3)             # [1. 1. 1.]
ones_2d = np.ones((2, 3))     # 2x3 array of ones

# Full (custom value)
full = np.full(4, 7)          # [7 7 7 7]
full_2d = np.full((2, 3), 5)  # 2x3 array filled with 5

# Empty (uninitialized)
empty = np.empty(3)           # Random values

# Identity matrix
identity = np.eye(3)          # 3x3 identity matrix
identity_rect = np.eye(3, 4)  # 3x4 identity matrix

# Range arrays
arange = np.arange(10)        # [0 1 2 3 4 5 6 7 8 9]
arange_step = np.arange(2, 10, 2)  # [2 4 6 8]
linspace = np.linspace(0, 1, 5)    # [0. 0.25 0.5 0.75 1.]
logspace = np.logspace(0, 2, 3)    # [1. 10. 100.]

# Random arrays
random = np.random.random(5)        # Random floats [0, 1)
randint = np.random.randint(0, 10, 5)  # Random integers
randn = np.random.randn(3, 3)       # Standard normal distribution
```

## 3. Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Shape and dimensions
print(arr.shape)      # (2, 3)
print(arr.ndim)       # 2
print(arr.size)       # 6
print(len(arr))       # 2 (length of first dimension)

# Data type
print(arr.dtype)      # int64 (or int32 on some systems)
print(arr.itemsize)   # 8 (bytes per element)
print(arr.nbytes)     # 48 (total bytes)

# Memory layout
print(arr.flags)      # Memory layout info
```

## 4. Array Indexing and Slicing

### Basic Indexing
```python
arr = np.array([0, 1, 2, 3, 4, 5])

# Single element
print(arr[0])         # 0
print(arr[-1])        # 5

# Slicing
print(arr[1:4])       # [1 2 3]
print(arr[::2])       # [0 2 4]
print(arr[::-1])      # [5 4 3 2 1 0]
```

### Multi-dimensional Indexing
```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Element access
print(arr_2d[0, 1])   # 2
print(arr_2d[1][2])   # 6

# Row/column access
print(arr_2d[0])      # [1 2 3] (first row)
print(arr_2d[:, 1])   # [2 5 8] (second column)

# Slicing
print(arr_2d[0:2, 1:3])  # [[2 3], [5 6]]
```

### Boolean Indexing
```python
arr = np.array([1, 2, 3, 4, 5])

# Boolean mask
mask = arr > 3
print(arr[mask])      # [4 5]
print(arr[arr > 3])   # [4 5] (direct)

# Multiple conditions
print(arr[(arr > 2) & (arr < 5)])  # [3 4]
```

### Fancy Indexing
```python
arr = np.array([10, 20, 30, 40, 50])

# Index arrays
indices = [0, 2, 4]
print(arr[indices])   # [10 30 50]

# 2D fancy indexing
arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
print(arr_2d[[0, 2], [1, 0]])  # [2 5]
```

## 5. Array Reshaping and Manipulation

### Reshaping
```python
arr = np.arange(12)

# Reshape
reshaped = arr.reshape(3, 4)
reshaped_3d = arr.reshape(2, 2, 3)

# Automatic dimension
auto_reshape = arr.reshape(-1, 4)  # (-1 means "figure it out")

# Flatten
flattened = reshaped.flatten()     # Returns copy
raveled = reshaped.ravel()         # Returns view if possible
```

### Dimension Manipulation
```python
arr = np.array([1, 2, 3])

# Add dimensions
expanded = np.expand_dims(arr, axis=0)  # [[1 2 3]]
newaxis = arr[np.newaxis, :]            # Same as above

# Remove dimensions
squeezed = np.squeeze(expanded)         # [1 2 3]

# Transpose
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr_2d.T                   # [[1 4], [2 5], [3 6]]
transposed_func = np.transpose(arr_2d)  # Same as above
```

### Joining and Splitting
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenation
concat = np.concatenate([arr1, arr2])   # [1 2 3 4 5 6]
vstack = np.vstack([arr1, arr2])        # [[1 2 3], [4 5 6]]
hstack = np.hstack([arr1, arr2])        # [1 2 3 4 5 6]

# Column and row stack
arr_2d1 = np.array([[1, 2], [3, 4]])
arr_2d2 = np.array([[5, 6], [7, 8]])
column_stack = np.column_stack([arr1, arr2])  # [[1 4], [2 5], [3 6]]
row_stack = np.row_stack([arr_2d1, arr_2d2])  # Same as vstack

# Splitting
split_arr = np.split(np.arange(9), 3)   # [array([0, 1, 2]), ...]
hsplit_arr = np.hsplit(arr_2d1, 2)      # Split horizontally
vsplit_arr = np.vsplit(arr_2d1, 2)      # Split vertically
```

## 6. Mathematical Operations

### Basic Arithmetic
```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# Element-wise operations
add = arr1 + arr2          # [6 8 10 12]
subtract = arr1 - arr2     # [-4 -4 -4 -4]
multiply = arr1 * arr2     # [5 12 21 32]
divide = arr1 / arr2       # [0.2 0.33 0.43 0.5]
power = arr1 ** 2          # [1 4 9 16]
modulo = arr1 % 3          # [1 2 0 1]

# Broadcasting
scalar_add = arr1 + 10     # [11 12 13 14]
```

### Mathematical Functions
```python
arr = np.array([1, 4, 9, 16])

# Square root, exponential, logarithm
sqrt = np.sqrt(arr)        # [1. 2. 3. 4.]
exp = np.exp(arr1)         # [e^1, e^2, e^3, e^4]
log = np.log(arr)          # Natural log
log10 = np.log10(arr)      # Base 10 log
log2 = np.log2(arr)        # Base 2 log

# Trigonometric functions
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
sin = np.sin(angles)
cos = np.cos(angles)
tan = np.tan(angles)

# Inverse trig functions
arcsin = np.arcsin([0, 0.5, 1])
arccos = np.arccos([0, 0.5, 1])
arctan = np.arctan([0, 1, np.inf])

# Hyperbolic functions
sinh = np.sinh([0, 1, 2])
cosh = np.cosh([0, 1, 2])
tanh = np.tanh([0, 1, 2])

# Rounding
arr_float = np.array([1.2, 2.7, 3.1, 4.9])
rounded = np.round(arr_float)      # [1. 3. 3. 5.]
floor = np.floor(arr_float)        # [1. 2. 3. 4.]
ceil = np.ceil(arr_float)          # [2. 3. 4. 5.]
```

### Aggregate Functions
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Basic statistics
sum_all = np.sum(arr)              # 21
sum_axis0 = np.sum(arr, axis=0)    # [5 7 9]
sum_axis1 = np.sum(arr, axis=1)    # [6 15]

mean = np.mean(arr)                # 3.5
median = np.median(arr)            # 3.5
std = np.std(arr)                  # Standard deviation
var = np.var(arr)                  # Variance

# Min/max
min_val = np.min(arr)              # 1
max_val = np.max(arr)              # 6
argmin = np.argmin(arr)            # 0 (index of min)
argmax = np.argmax(arr)            # 5 (index of max)

# Cumulative operations
cumsum = np.cumsum(arr)            # [1 3 6 10 15 21]
cumprod = np.cumprod(arr)          # [1 2 6 24 120 720]
```

## 7. Linear Algebra

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Dot product
dot = np.dot(A, B)                 # [[19 22], [43 50]]
dot_operator = A @ B               # Same as above (Python 3.5+)

# Element-wise multiplication
element_wise = A * B               # [[5 12], [21 32]]

# Matrix operations
det = np.linalg.det(A)             # Determinant: -2.0
inv = np.linalg.inv(A)             # Inverse matrix
eigenvals, eigenvecs = np.linalg.eig(A)  # Eigenvalues and eigenvectors

# Solving linear systems (Ax = b)
b = np.array([1, 2])
x = np.linalg.solve(A, b)          # Solution to Ax = b

# SVD (Singular Value Decomposition)
U, s, Vt = np.linalg.svd(A)

# Norms
norm_2 = np.linalg.norm(A)         # Frobenius norm
norm_1 = np.linalg.norm(A, ord=1)  # 1-norm
```

## 8. Array Comparison and Logic

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([1, 3, 3, 2])

# Element-wise comparison
equal = arr1 == arr2               # [True False True False]
not_equal = arr1 != arr2           # [False True False True]
greater = arr1 > arr2              # [False False False True]
less = arr1 < arr2                 # [False True False False]

# Logical operations
logical_and = np.logical_and(arr1 > 2, arr2 < 4)
logical_or = np.logical_or(arr1 > 3, arr2 < 2)
logical_not = np.logical_not(arr1 > 2)

# Array-wise comparisons
array_equal = np.array_equal(arr1, arr2)    # False
allclose = np.allclose(arr1, arr2, atol=1)  # True (within tolerance)

# Any/all
any_greater = np.any(arr1 > 3)     # True
all_positive = np.all(arr1 > 0)    # True
```

## 9. Conditional Operations

```python
arr = np.array([1, 2, 3, 4, 5])

# Where function
result = np.where(arr > 3, arr, 0)     # [0 0 0 4 5]
result = np.where(arr > 3, 'big', 'small')  # Conditional replacement

# Select and choose
conditions = [arr < 2, arr > 4]
choices = ['small', 'large']
result = np.select(conditions, choices, default='medium')

# Clip (limit values)
clipped = np.clip(arr, 2, 4)           # [2 2 3 4 4]
```

## 10. Sorting and Searching

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Sorting
sorted_arr = np.sort(arr)              # [1 1 2 3 4 5 6 9]
sort_indices = np.argsort(arr)         # [1 3 6 0 2 4 7 5]

# Partial sorting
partition = np.partition(arr, 3)       # Partially sorted

# Searching
index = np.searchsorted(sorted_arr, 4) # 4 (insertion point)
indices = np.where(arr == 1)           # (array([1, 3]),)

# Unique values
unique_vals = np.unique(arr)           # [1 2 3 4 5 6 9]
unique_vals, counts = np.unique(arr, return_counts=True)
```

## 11. Set Operations

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

# Set operations
intersection = np.intersect1d(arr1, arr2)    # [3 4]
union = np.union1d(arr1, arr2)               # [1 2 3 4 5 6]
difference = np.setdiff1d(arr1, arr2)        # [1 2]
symmetric_diff = np.setxor1d(arr1, arr2)     # [1 2 5 6]

# Membership test
in_arr = np.in1d(arr1, arr2)                 # [False False True True]
```

## 12. Input/Output

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save and load (binary format)
np.save('array.npy', arr)
loaded_arr = np.load('array.npy')

# Save and load (text format)
np.savetxt('array.txt', arr, delimiter=',')
loaded_text = np.loadtxt('array.txt', delimiter=',')

# CSV files
np.savetxt('array.csv', arr, delimiter=',', header='col1,col2,col3')
csv_data = np.loadtxt('array.csv', delimiter=',', skiprows=1)
```

## 13. Random Number Generation

```python
# Set seed for reproducibility
np.random.seed(42)

# Random sampling
random_floats = np.random.random(5)           # [0, 1)
random_normal = np.random.normal(0, 1, 5)     # Normal distribution
random_ints = np.random.randint(1, 11, 5)     # Random integers

# Random choice
arr = np.array([1, 2, 3, 4, 5])
choice = np.random.choice(arr, 3)             # Random sample
choice_prob = np.random.choice(arr, 3, p=[0.1, 0.1, 0.1, 0.1, 0.6])

# Shuffle
np.random.shuffle(arr)                        # In-place shuffle
permuted = np.random.permutation(arr)         # Returns shuffled copy

# Distributions
uniform = np.random.uniform(0, 10, 5)         # Uniform distribution
exponential = np.random.exponential(2, 5)     # Exponential distribution
binomial = np.random.binomial(10, 0.5, 5)     # Binomial distribution
```

## 14. Broadcasting

```python
# Broadcasting rules: NumPy automatically broadcasts arrays with different shapes

# Scalar with array
arr = np.array([1, 2, 3, 4])
result = arr + 10                             # [11 12 13 14]

# Different shaped arrays
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])
result = arr_2d + arr_1d                      # Broadcasting along axis 1

# Manual broadcasting
broadcasted = np.broadcast_arrays(arr_2d, arr_1d)
```

## 15. Data Types

```python
# Common data types
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1, 2, 3], dtype=np.float64)
bool_arr = np.array([True, False, True], dtype=np.bool_)
string_arr = np.array(['a', 'b', 'c'], dtype='U1')

# Type conversion
float_from_int = int_arr.astype(np.float64)
int_from_float = np.array([1.1, 2.9]).astype(int)

# Complex numbers
complex_arr = np.array([1+2j, 3+4j])
real_part = complex_arr.real
imag_part = complex_arr.imag
```

## 16. Memory and Performance

```python
# Views vs copies
arr = np.array([1, 2, 3, 4])
view = arr[::2]        # Creates a view
copy = arr.copy()      # Creates a copy

# Check if array owns its data
print(view.base is arr)    # True (view)
print(copy.base is None)   # True (copy)

# Memory layout
arr_c = np.array([[1, 2], [3, 4]], order='C')  # C-contiguous
arr_f = np.array([[1, 2], [3, 4]], order='F')  # Fortran-contiguous

# Memory usage
print(arr.nbytes)      # Memory usage in bytes
```

## 17. Useful Utility Functions

```python
# Array information
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.info(arr)           # Detailed array information

# Array testing
np.isfinite([1, np.inf, np.nan])     # [True False False]
np.isnan([1, np.nan, 3])             # [False True False]
np.isinf([1, np.inf, 3])             # [False True False]

# Array manipulation
arr_flat = arr.flat                   # Flat iterator
diag = np.diag([1, 2, 3])            # Diagonal matrix
trace = np.trace(arr)                 # Sum of diagonal elements

# Meshgrid for coordinate arrays
x = np.array([1, 2, 3])
y = np.array([4, 5])
X, Y = np.meshgrid(x, y)
```

## 18. Advanced Indexing Examples

```python
# Multi-dimensional boolean indexing
arr_3d = np.random.randint(0, 10, (3, 3, 3))
mask = arr_3d > 5
filtered = arr_3d[mask]

# Index arrays for fancy indexing
rows = np.array([0, 1, 2])
cols = np.array([2, 1, 0])
arr_2d = np.arange(9).reshape(3, 3)
selected = arr_2d[rows, cols]         # Elements at (0,2), (1,1), (2,0)

# Using ix_ for cross-product indexing
selected_block = arr_2d[np.ix_([0, 2], [1, 2])]
```

## Key Tips and Best Practices

1. **Use vectorized operations** instead of Python loops for better performance
2. **Understand broadcasting** to work efficiently with different array shapes
3. **Be aware of views vs copies** to manage memory effectively
4. **Use appropriate data types** to optimize memory usage
5. **Leverage NumPy's built-in functions** rather than implementing your own
6. **Use axis parameter** in reduction operations to control which dimension to reduce
7. **Profile your code** to identify bottlenecks when working with large arrays

This cheat sheet covers the most commonly used NumPy functions and operations. NumPy is extremely powerful and has many more specialized functions for specific use cases!