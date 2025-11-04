---
applyTo: '**'
---

# Performance Optimization Guidelines

## When Performance Matters

### Optimize For:
- **Large datasets**: >100K rows or >1GB memory
- **Repeated operations**: Functions called 1000+ times
- **Production pipelines**: Code that runs regularly
- **Week 10 projects**: Where efficiency is expected
- **Resource constraints**: Limited memory or CPU

### Don't Optimize:
- **Week 1-4 exercises**: Clarity over performance
- **One-off scripts**: Run once and done
- **Small datasets**: <1000 rows
- **Prototyping**: Get it working first
- **Already fast enough**: <1 second total runtime

## Vectorized Operations vs Loops

### Use Vectorization (NumPy/Pandas)
**Slow (loops):**
```python
# Calculate square of each element
result = []
for value in values:
    result.append(value ** 2)
```

**Fast (vectorized):**
```python
# NumPy vectorization (~100x faster)
result = values ** 2

# Pandas vectorization
df['squared'] = df['values'] ** 2
```

### Common Vectorization Patterns

**Element-wise operations:**
```python
# Bad
for i in range(len(df)):
    df.loc[i, 'result'] = df.loc[i, 'a'] + df.loc[i, 'b']

# Good
df['result'] = df['a'] + df['b']
```

**Conditional operations:**
```python
# Bad
for i in range(len(df)):
    if df.loc[i, 'value'] > 0:
        df.loc[i, 'category'] = 'positive'
    else:
        df.loc[i, 'category'] = 'negative'

# Good
df['category'] = np.where(df['value'] > 0, 'positive', 'negative')
```

**Multiple conditions:**
```python
# Bad
categories = []
for value in df['score']:
    if value >= 90:
        categories.append('A')
    elif value >= 80:
        categories.append('B')
    else:
        categories.append('C')

# Good
conditions = [
    (df['score'] >= 90),
    (df['score'] >= 80),
]
choices = ['A', 'B']
df['grade'] = np.select(conditions, choices, default='C')
```

## Memory-Efficient Pandas

### Data Loading Optimization

**Select columns early:**
```python
# Bad - Load everything then select
df = pd.read_csv('large_file.csv')
df = df[['col1', 'col2']]

# Good - Select during load
df = pd.read_csv('large_file.csv', usecols=['col1', 'col2'])
```

**Specify dtypes:**
```python
# Bad - Pandas infers types (may use more memory)
df = pd.read_csv('file.csv')

# Good - Specify efficient types
dtypes = {
    'id': 'int32',           # int64 -> int32 saves 50% memory
    'category': 'category',  # object -> category saves memory for repeating values
    'value': 'float32'       # float64 -> float32 saves 50% memory
}
df = pd.read_csv('file.csv', dtype=dtypes)
```

**Chunk processing:**
```python
# Bad - Load entire file into memory
df = pd.read_csv('huge_file.csv')
result = df['column'].sum()

# Good - Process in chunks
result = 0
for chunk in pd.read_csv('huge_file.csv', chunksize=10000):
    result += chunk['column'].sum()
```

### DataFrame Memory Optimization

**Convert to category:**
```python
# For columns with repeating values (<50% unique)
# Bad - Object dtype uses lots of memory
df['status'] = df['status']  # object dtype

# Good - Category dtype
df['status'] = df['status'].astype('category')

# Memory savings example:
# 1M rows with 5 unique values:
# object: ~8MB, category: ~1MB (8x reduction)
```

**Downcast numeric types:**
```python
# Bad - Default int64/float64
df['age'] = df['age']  # int64 (8 bytes per value)

# Good - Downcast to smallest type
df['age'] = pd.to_numeric(df['age'], downcast='integer')  # int8 (1 byte)
df['price'] = pd.to_numeric(df['price'], downcast='float')
```

**Check memory usage:**
```python
# See memory usage per column
print(df.memory_usage(deep=True))

# Total memory usage
print(f"Total: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Avoid Memory Copies

**Use inplace carefully:**
```python
# Creates copy (2x memory temporarily)
df = df.dropna()
df = df.sort_values('column')

# In-place operations (no copy, but breaks chaining)
df.dropna(inplace=True)
df.sort_values('column', inplace=True)

# Balance: Chain operations without excessive copies
df = (df
      .dropna()
      .sort_values('column')
      .reset_index(drop=True))
```

**Delete when done:**
```python
# Free memory explicitly
df_temp = df.copy()
# ... use df_temp ...
del df_temp

# Or use in limited scope
def process_subset():
    df_temp = df.head(1000)
    # ... process ...
    return result
# df_temp is freed when function exits
```

## Subsampling Strategies

### When to Subsample
- Dataset too large for memory
- Model training taking too long
- Exploratory analysis phase
- Development/testing

### Stratified Sampling
```python
# Maintain class distribution
from sklearn.model_selection import train_test_split

# Sample 10% while preserving class balance
df_sample, _ = train_test_split(
    df,
    train_size=0.1,
    random_state=42,
    stratify=df['target']
)
```

### Random Sampling
```python
# Simple random sample
df_sample = df.sample(n=10000, random_state=42)

# Or by fraction
df_sample = df.sample(frac=0.1, random_state=42)
```

### CLI Argument Pattern
```python
# Allow user to control sample size
parser.add_argument('--train-sample-size', type=int, default=None,
                    help='Subsample training data to N rows for faster testing')

if args.train_sample_size and len(df) > args.train_sample_size:
    df = df.sample(n=args.train_sample_size, random_state=RANDOM_STATE)
    logger.info(f"Subsampled to {args.train_sample_size} rows")
```

## Caching Intermediate Results

### Save Expensive Computations
```python
# Check if cached result exists
cache_path = Path('outputs/cache/preprocessed_data.pkl')

if cache_path.exists():
    # Load from cache
    df = pd.read_pickle(cache_path)
    logger.info("Loaded preprocessed data from cache")
else:
    # Compute and cache
    df = expensive_preprocessing(raw_df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    logger.info("Cached preprocessed data")
```

### Caching Patterns
```python
from functools import lru_cache

# Cache function results (for pure functions)
@lru_cache(maxsize=128)
def expensive_calculation(param):
    """Compute expensive result, cached for repeated calls."""
    # ... expensive operation ...
    return result

# Clear cache if needed
expensive_calculation.cache_clear()
```

## Performance vs Code Clarity

### Premature Optimization (Avoid)
```python
# Over-optimized, hard to read
df['result'] = (df['a'].values * df['b'].values + df['c'].values) / df['d'].values

# Better: Clear and still fast
df['product'] = df['a'] * df['b']
df['sum'] = df['product'] + df['c']
df['result'] = df['sum'] / df['d']
```

### When to Sacrifice Clarity
- Proven bottleneck (profiled)
- 10x+ performance improvement
- Still understandable with comments
- Performance is critical requirement

### Optimization Comment Pattern
```python
# Performance optimization: Use NumPy arrays directly instead of pandas
# to avoid overhead in tight loop. Benchmarked: 10x faster for N>10000.
values_array = df['values'].values
result = np.sum(values_array ** 2)  # Instead of df['values'].sum()
```

## Profiling and Benchmarking

### Simple Timing
```python
import time

start = time.perf_counter()
result = expensive_operation()
elapsed = time.perf_counter() - start
print(f"Operation took {elapsed:.2f} seconds")
```

### Function Timing Decorator
```python
def timeit(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

@timeit
def process_data(df):
    # ... processing ...
    return result
```

### Compare Approaches
```python
import timeit

# Compare two implementations
method1 = timeit.timeit(
    'df["result"] = df.apply(lambda row: row["a"] + row["b"], axis=1)',
    setup='import pandas as pd; df = pd.DataFrame({"a": range(1000), "b": range(1000)})',
    number=100
)

method2 = timeit.timeit(
    'df["result"] = df["a"] + df["b"]',
    setup='import pandas as pd; df = pd.DataFrame({"a": range(1000), "b": range(1000)})',
    number=100
)

print(f"Method 1: {method1:.4f}s")
print(f"Method 2: {method2:.4f}s")
print(f"Speedup: {method1/method2:.1f}x")
```

## Parallel Processing

### When to Parallelize
- Independent operations (no shared state)
- CPU-bound tasks (not I/O bound)
- Large enough workload (>1 second per task)
- Multiple cores available

### Simple Parallel Pattern
```python
from multiprocessing import Pool

def process_item(item):
    """Process single item (must be pickleable)."""
    # ... processing ...
    return result

# Parallel processing
with Pool(processes=4) as pool:
    results = pool.map(process_item, items)
```

### When NOT to Parallelize
- Small datasets (overhead > benefit)
- I/O bound operations (file reading)
- Shared state complications
- Debugging difficulty outweighs benefit

## Optimization Checklist

Before optimizing:
1. ✅ **Profile first**: Identify actual bottleneck
2. ✅ **Measure baseline**: Time current implementation
3. ✅ **Set target**: How fast does it need to be?
4. ✅ **Optimize bottleneck**: Focus on slowest part
5. ✅ **Measure improvement**: Verify speedup
6. ✅ **Maintain correctness**: Test results unchanged
7. ✅ **Document changes**: Comment why optimization was needed

Quick wins (try first):
- Use vectorization instead of loops
- Specify dtypes when loading data
- Select only needed columns
- Convert to category dtype for repeated values
- Cache expensive computations
