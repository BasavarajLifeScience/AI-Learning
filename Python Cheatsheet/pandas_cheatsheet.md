# Complete Pandas Cheat Sheet with Examples

## 1. Import and Setup

```python
import pandas as pd
import numpy as np

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
```

## 2. Creating DataFrames and Series

### Creating Series
```python
# From list
s = pd.Series([1, 2, 3, 4, 5])

# From dictionary
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# With custom index
s = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
```

### Creating DataFrames
```python
# From dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# From list of dictionaries
df = pd.DataFrame([
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
])

# From numpy array
df = pd.DataFrame(np.random.randn(4, 3), 
                  columns=['A', 'B', 'C'])

# Empty DataFrame
df = pd.DataFrame()
```

## 3. Reading and Writing Data

### Reading Data
```python
# CSV files
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', sep=';', encoding='utf-8')

# Excel files
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# JSON files
df = pd.read_json('file.json')

# SQL databases
df = pd.read_sql('SELECT * FROM table', connection)

# HTML tables
df = pd.read_html('url_or_file.html')[0]

# Parquet files
df = pd.read_parquet('file.parquet')
```

### Writing Data
```python
# CSV
df.to_csv('output.csv', index=False)

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# JSON
df.to_json('output.json', orient='records')

# SQL
df.to_sql('table_name', connection, if_exists='replace')

# Parquet
df.to_parquet('output.parquet')
```

## 4. Data Inspection

```python
# Basic info
df.head()           # First 5 rows
df.tail(10)         # Last 10 rows
df.shape            # (rows, columns)
df.info()           # Data types and memory usage
df.describe()       # Statistical summary
df.dtypes           # Data types of each column

# Missing values
df.isnull().sum()   # Count of null values per column
df.notnull().sum()  # Count of non-null values per column

# Unique values
df['column'].unique()           # Unique values in column
df['column'].nunique()          # Number of unique values
df['column'].value_counts()     # Count of each unique value

# Memory usage
df.memory_usage(deep=True)
```

## 5. Data Selection and Indexing

### Column Selection
```python
# Single column
df['name']
df.name  # Dot notation (if column name is valid identifier)

# Multiple columns
df[['name', 'age']]

# All columns except specific ones
df.drop(['name'], axis=1)
```

### Row Selection
```python
# By index position
df.iloc[0]          # First row
df.iloc[0:3]        # First 3 rows
df.iloc[-1]         # Last row

# By index label
df.loc[0]           # Row with index 0
df.loc[0:2]         # Rows with index 0 to 2 (inclusive)

# Boolean indexing
df[df['age'] > 25]
df[df['name'].str.contains('A')]
```

### Combined Selection
```python
# Specific rows and columns
df.loc[0:2, 'name':'age']
df.iloc[0:3, 1:3]

# Boolean indexing with specific columns
df.loc[df['age'] > 25, ['name', 'city']]
```

## 6. Data Filtering

```python
# Single condition
df[df['age'] > 25]
df[df['name'] == 'Alice']

# Multiple conditions
df[(df['age'] > 25) & (df['city'] == 'NYC')]
df[(df['age'] < 25) | (df['age'] > 35)]

# String methods
df[df['name'].str.startswith('A')]
df[df['name'].str.contains('ice')]
df[df['name'].str.len() > 3]

# isin() method
df[df['city'].isin(['NYC', 'LA'])]

# between() method
df[df['age'].between(25, 35)]

# query() method
df.query('age > 25 and city == "NYC"')
```

## 7. Data Cleaning

### Handling Missing Values
```python
# Check for missing values
df.isnull()
df.isna()

# Drop missing values
df.dropna()                    # Drop rows with any NaN
df.dropna(axis=1)             # Drop columns with any NaN
df.dropna(thresh=2)           # Keep rows with at least 2 non-NaN values

# Fill missing values
df.fillna(0)                  # Fill with 0
df.fillna(method='ffill')     # Forward fill
df.fillna(method='bfill')     # Backward fill
df.fillna(df.mean())          # Fill with mean

# Fill specific columns
df['age'].fillna(df['age'].mean(), inplace=True)
```

### Removing Duplicates
```python
# Check for duplicates
df.duplicated()
df.duplicated(subset=['name'])

# Remove duplicates
df.drop_duplicates()
df.drop_duplicates(subset=['name'], keep='first')
```

### Data Type Conversion
```python
# Convert data types
df['age'] = df['age'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')

# Convert to numeric (coerce errors to NaN)
df['numeric'] = pd.to_numeric(df['column'], errors='coerce')
```

## 8. Data Transformation

### Adding/Modifying Columns
```python
# Add new column
df['age_squared'] = df['age'] ** 2
df['full_name'] = df['first_name'] + ' ' + df['last_name']

# Conditional column creation
df['age_group'] = np.where(df['age'] < 30, 'Young', 'Old')

# Using apply()
df['age_plus_10'] = df['age'].apply(lambda x: x + 10)

# Using map() for Series
df['grade'] = df['score'].map({90: 'A', 80: 'B', 70: 'C'})
```

### String Operations
```python
# String methods
df['name'].str.lower()
df['name'].str.upper()
df['name'].str.title()
df['name'].str.strip()
df['name'].str.replace('old', 'new')
df['name'].str.split(' ')
df['name'].str.extract('([A-Z])')  # Extract using regex
```

### DateTime Operations
```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()

# Date arithmetic
df['days_ago'] = (pd.Timestamp.now() - df['date']).dt.days
```

## 9. Grouping and Aggregation

### Basic Grouping
```python
# Group by single column
grouped = df.groupby('city')
grouped.mean()
grouped.sum()
grouped.count()

# Group by multiple columns
df.groupby(['city', 'age_group']).mean()

# Aggregate specific columns
df.groupby('city')['age'].mean()
df.groupby('city')[['age', 'salary']].mean()
```

### Advanced Aggregation
```python
# Multiple aggregation functions
df.groupby('city').agg({
    'age': ['mean', 'std'],
    'salary': ['sum', 'count']
})

# Custom aggregation functions
df.groupby('city')['age'].agg(['mean', 'std', lambda x: x.max() - x.min()])

# Apply custom function to groups
df.groupby('city').apply(lambda x: x.age.max() - x.age.min())
```

### Pivot Tables
```python
# Basic pivot table
pivot = df.pivot_table(values='salary', 
                      index='city', 
                      columns='department', 
                      aggfunc='mean')

# Multiple values and functions
pivot = df.pivot_table(values=['salary', 'age'],
                      index='city',
                      columns='department',
                      aggfunc={'salary': 'mean', 'age': 'count'})
```

## 10. Sorting

```python
# Sort by single column
df.sort_values('age')
df.sort_values('age', ascending=False)

# Sort by multiple columns
df.sort_values(['city', 'age'])
df.sort_values(['city', 'age'], ascending=[True, False])

# Sort by index
df.sort_index()
df.sort_index(ascending=False)
```

## 11. Merging and Joining

### Concatenation
```python
# Concatenate DataFrames
pd.concat([df1, df2])                    # Vertical
pd.concat([df1, df2], axis=1)           # Horizontal
pd.concat([df1, df2], ignore_index=True) # Reset index
```

### Merging
```python
# Inner join (default)
merged = pd.merge(df1, df2, on='key')

# Different join types
merged = pd.merge(df1, df2, on='key', how='left')
merged = pd.merge(df1, df2, on='key', how='right')
merged = pd.merge(df1, df2, on='key', how='outer')

# Merge on multiple columns
merged = pd.merge(df1, df2, on=['key1', 'key2'])

# Merge with different column names
merged = pd.merge(df1, df2, left_on='key1', right_on='key2')
```

### Join (Index-based)
```python
# Join on index
df1.join(df2)
df1.join(df2, how='outer')
df1.join(df2, rsuffix='_right')
```

## 12. Reshaping Data

### Melting (Wide to Long)
```python
# Basic melt
melted = pd.melt(df, id_vars=['name'], value_vars=['math', 'science'])

# Melt with custom names
melted = pd.melt(df, 
                id_vars=['name'],
                value_vars=['math', 'science'],
                var_name='subject',
                value_name='score')
```

### Pivoting (Long to Wide)
```python
# Basic pivot
pivoted = df.pivot(index='name', columns='subject', values='score')

# Pivot with multiple values
pivoted = df.pivot(index='name', columns='subject', values=['score', 'grade'])
```

### Stack/Unstack
```python
# Stack (columns to rows)
stacked = df.stack()

# Unstack (rows to columns)
unstacked = df.unstack()
unstacked = df.unstack(level=0)  # Specify level
```

## 13. Statistical Operations

```python
# Basic statistics
df.mean()           # Mean
df.median()         # Median
df.std()            # Standard deviation
df.var()            # Variance
df.min()            # Minimum
df.max()            # Maximum
df.sum()            # Sum
df.count()          # Count of non-null values

# Quantiles
df.quantile(0.25)   # 25th percentile
df.quantile([0.25, 0.5, 0.75])  # Multiple quantiles

# Correlation
df.corr()           # Correlation matrix
df['col1'].corr(df['col2'])  # Correlation between two columns

# Rolling statistics
df['price'].rolling(window=30).mean()    # 30-day moving average
df['price'].rolling(window=30).std()     # 30-day rolling std
```

## 14. Advanced Operations

### Apply Functions
```python
# Apply to entire DataFrame
df.apply(np.sum)              # Sum each column
df.apply(np.sum, axis=1)      # Sum each row

# Apply to specific column
df['age'].apply(lambda x: x * 2)

# Apply with multiple columns
df.apply(lambda row: row['age'] + row['salary'], axis=1)

# Map function to Series
df['grade'] = df['score'].map({90: 'A', 80: 'B', 70: 'C'})
```

### Window Functions
```python
# Cumulative operations
df['cumsum'] = df['value'].cumsum()
df['cumprod'] = df['value'].cumprod()
df['cummax'] = df['value'].cummax()

# Shift operations
df['prev_value'] = df['value'].shift(1)
df['next_value'] = df['value'].shift(-1)

# Percentage change
df['pct_change'] = df['value'].pct_change()
```

### MultiIndex Operations
```python
# Create MultiIndex
df.set_index(['col1', 'col2'], inplace=True)

# Access MultiIndex levels
df.loc['level1_value']
df.loc[('level1_value', 'level2_value')]

# Reset MultiIndex
df.reset_index()

# Swap index levels
df.swaplevel(0, 1)
```

## 15. Time Series Operations

```python
# Create date range
dates = pd.date_range('2023-01-01', periods=100, freq='D')

# Set datetime index
df.set_index('date', inplace=True)

# Resample time series
df.resample('M').mean()      # Monthly average
df.resample('W').sum()       # Weekly sum

# Time-based selection
df['2023']                   # All data from 2023
df['2023-01':'2023-03']     # January to March 2023

# Time zone operations
df.tz_localize('UTC')
df.tz_convert('US/Eastern')
```

## 16. Performance Tips

```python
# Use vectorized operations instead of loops
df['new_col'] = df['col1'] + df['col2']  # Fast
# Instead of: df['new_col'] = df.apply(lambda x: x['col1'] + x['col2'], axis=1)

# Use .loc for setting values
df.loc[df['age'] > 30, 'category'] = 'Senior'

# Use categorical data for strings with few unique values
df['category'] = df['category'].astype('category')

# Use query() for complex filtering
df.query('age > 30 and salary < 50000')

# Chain operations
result = (df
          .groupby('category')
          .agg({'salary': 'mean', 'age': 'count'})
          .reset_index()
          .sort_values('salary', ascending=False)
         )
```

## 17. Common Patterns and Examples

### Data Cleaning Pipeline
```python
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['salary'].fillna(df['salary'].mean(), inplace=True)
    
    # Fix data types
    df['date'] = pd.to_datetime(df['date'])
    df['category'] = df['category'].astype('category')
    
    # Remove outliers (example: age > 100)
    df = df[df['age'] <= 100]
    
    return df
```

### Complex Aggregation Example
```python
# Sales analysis example
sales_summary = (df
    .groupby(['region', 'product'])
    .agg({
        'sales': ['sum', 'mean', 'count'],
        'profit': 'sum',
        'customer_id': 'nunique'
    })
    .round(2)
    .reset_index()
)
```

### Creating Age Groups
```python
def categorize_age(age):
    if age < 18:
        return 'Minor'
    elif age < 30:
        return 'Young Adult'
    elif age < 50:
        return 'Adult'
    else:
        return 'Senior'

df['age_group'] = df['age'].apply(categorize_age)
# Or using cut()
df['age_group'] = pd.cut(df['age'], 
                        bins=[0, 18, 30, 50, 100], 
                        labels=['Minor', 'Young Adult', 'Adult', 'Senior'])
```

This cheat sheet covers the most important pandas operations you'll use in data analysis. Remember to practice these operations with real datasets to become proficient!