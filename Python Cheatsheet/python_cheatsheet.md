# Python Cheatsheet

## Basic Syntax

### Variables and Data Types
```python
# Variables (no declaration needed)
name = "Alice"
age = 25
height = 5.6
is_student = True

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0
```

### Data Types
```python
# Numbers
integer = 42
float_num = 3.14
complex_num = 3 + 4j

# Strings
single_quote = 'Hello'
double_quote = "World"
multi_line = """This is a
multi-line string"""
f_string = f"Hello {name}, you are {age} years old"

# Boolean
is_true = True
is_false = False
```

## Operators

### Arithmetic
```python
+   # Addition
-   # Subtraction
*   # Multiplication
/   # Division (float result)
//  # Floor division
%   # Modulus
**  # Exponentiation
```

### Comparison
```python
==  # Equal
!=  # Not equal
<   # Less than
>   # Greater than
<=  # Less than or equal
>=  # Greater than or equal
```

### Logical
```python
and  # Logical AND
or   # Logical OR
not  # Logical NOT
```

## Data Structures

### Lists (Mutable)
```python
# Creation
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# Access
first_fruit = fruits[0]    # "apple"
last_fruit = fruits[-1]    # "cherry"
slice_fruits = fruits[1:3] # ["banana", "cherry"]

# Methods
fruits.append("orange")        # Add to end
fruits.insert(1, "grape")     # Insert at index
fruits.remove("banana")       # Remove first occurrence
popped = fruits.pop()         # Remove and return last
fruits.sort()                 # Sort in place
fruits.reverse()              # Reverse in place
length = len(fruits)          # Get length
```

### Tuples (Immutable)
```python
# Creation
coordinates = (3, 4)
single_item = (42,)  # Note the comma
empty_tuple = ()

# Access (same as lists)
x = coordinates[0]
y = coordinates[1]

# Unpacking
x, y = coordinates
```

### Dictionaries
```python
# Creation
person = {"name": "Alice", "age": 25, "city": "NYC"}
empty_dict = {}

# Access
name = person["name"]           # Raises KeyError if not found
age = person.get("age", 0)      # Returns 0 if not found

# Modification
person["age"] = 26              # Update
person["job"] = "Engineer"      # Add new key

# Methods
keys = person.keys()            # Get all keys
values = person.values()        # Get all values
items = person.items()          # Get key-value pairs
```

### Sets
```python
# Creation
numbers = {1, 2, 3, 4, 5}
empty_set = set()

# Methods
numbers.add(6)                  # Add element
numbers.remove(3)               # Remove (raises error if not found)
numbers.discard(10)             # Remove (no error if not found)

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union = set1 | set2             # {1, 2, 3, 4, 5}
intersection = set1 & set2      # {3}
difference = set1 - set2        # {1, 2}
```

## Control Flow

### Conditionals
```python
if condition:
    # code block
elif another_condition:
    # code block
else:
    # code block

# Ternary operator
result = value_if_true if condition else value_if_false
```

### Loops
```python
# For loop
for item in iterable:
    print(item)

# For loop with index
for i, item in enumerate(iterable):
    print(f"{i}: {item}")

# For loop with range
for i in range(5):        # 0 to 4
    print(i)

for i in range(1, 6):     # 1 to 5
    print(i)

for i in range(0, 10, 2): # 0, 2, 4, 6, 8
    print(i)

# While loop
while condition:
    # code block
    
# Loop control
break       # Exit loop
continue    # Skip to next iteration
```

## Functions

### Basic Functions
```python
def greet(name):
    return f"Hello, {name}!"

def add(a, b=0):  # Default parameter
    return a + b

def multiply(*args):  # Variable arguments
    result = 1
    for num in args:
        result *= num
    return result

def person_info(**kwargs):  # Keyword arguments
    for key, value in kwargs.items():
        print(f"{key}: {value}")
```

### Lambda Functions
```python
# Lambda (anonymous) functions
square = lambda x: x ** 2
add = lambda x, y: x + y

# Common use with map, filter, sort
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

## List Comprehensions
```python
# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in range(10)]

# With condition: [expression for item in iterable if condition]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested loops
matrix = [[i*j for j in range(3)] for i in range(3)]

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["hello", "world"]}

# Set comprehension
unique_lengths = {len(word) for word in ["hello", "world", "hi"]}
```

## Exception Handling
```python
try:
    risky_operation()
except SpecificError as e:
    print(f"Specific error: {e}")
except Exception as e:
    print(f"General error: {e}")
else:
    print("No errors occurred")
finally:
    print("This always runs")

# Raising exceptions
raise ValueError("Something went wrong")
```

## File Operations
```python
# Reading files
with open("file.txt", "r") as f:
    content = f.read()          # Read entire file
    
with open("file.txt", "r") as f:
    lines = f.readlines()       # Read all lines as list

with open("file.txt", "r") as f:
    for line in f:              # Read line by line
        print(line.strip())

# Writing files
with open("file.txt", "w") as f:
    f.write("Hello, World!")

with open("file.txt", "a") as f:  # Append mode
    f.write("Additional text")
```

## Classes and Objects
```python
class Person:
    # Class variable
    species = "Homo sapiens"
    
    def __init__(self, name, age):  # Constructor
        self.name = name            # Instance variable
        self.age = age
    
    def greet(self):               # Instance method
        return f"Hi, I'm {self.name}"
    
    @classmethod
    def from_string(cls, person_str):  # Class method
        name, age = person_str.split("-")
        return cls(name, int(age))
    
    @staticmethod
    def is_adult(age):             # Static method
        return age >= 18

# Usage
person = Person("Alice", 25)
print(person.greet())
adult_check = Person.is_adult(25)
```

## Common Built-in Functions
```python
# Type conversion
int("123")      # String to integer
float("3.14")   # String to float
str(123)        # Integer to string
list("hello")   # String to list of characters

# Math functions
abs(-5)         # Absolute value
min(1, 2, 3)    # Minimum value
max(1, 2, 3)    # Maximum value
sum([1, 2, 3])  # Sum of iterable
round(3.14159, 2)  # Round to 2 decimal places

# Sequence functions
len([1, 2, 3])     # Length
sorted([3, 1, 2])  # Return sorted list
reversed([1, 2, 3]) # Return reversed iterator
enumerate(['a', 'b', 'c'])  # Return index, value pairs
zip([1, 2], ['a', 'b'])     # Combine iterables

# Type checking
type(42)           # <class 'int'>
isinstance(42, int) # True
```

## String Methods
```python
text = "Hello, World!"

# Case methods
text.upper()        # "HELLO, WORLD!"
text.lower()        # "hello, world!"
text.title()        # "Hello, World!"
text.capitalize()   # "Hello, world!"

# Search and replace
text.find("World")     # Returns index (7) or -1
"World" in text        # True
text.replace("World", "Python")  # "Hello, Python!"

# Split and join
"a,b,c".split(",")     # ["a", "b", "c"]
",".join(["a", "b", "c"])  # "a,b,c"

# Whitespace
"  hello  ".strip()    # "hello"
"  hello  ".lstrip()   # "hello  "
"  hello  ".rstrip()   # "  hello"

# Checking methods
text.startswith("Hello")  # True
text.endswith("!")        # True
"123".isdigit()          # True
"abc".isalpha()          # True
```

## Useful Modules to Import
```python
import os           # Operating system interface
import sys          # System-specific parameters
import math         # Mathematical functions
import random       # Random number generation
import datetime     # Date and time handling
import json         # JSON encoder/decoder
import re           # Regular expressions
from collections import defaultdict, Counter
```

## Quick Tips
```python
# Swap variables
a, b = b, a

# Check if list is empty
if not my_list:
    print("List is empty")

# Get unique items while preserving order
unique_items = list(dict.fromkeys(items))

# Flatten a list of lists
flat_list = [item for sublist in nested_list for item in sublist]

# Check multiple conditions
if any([condition1, condition2, condition3]):
    print("At least one condition is true")

if all([condition1, condition2, condition3]):
    print("All conditions are true")
```