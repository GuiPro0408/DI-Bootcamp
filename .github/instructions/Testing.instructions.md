---
applyTo: '**/*.py'
---

# Testing & Validation Guidelines

## Manual Test Scenarios

### Exercise Scripts (Week 1-4)
For each exercise, verify:
1. **Happy path**: Works with valid, typical input
2. **Edge cases**: Empty input, boundary values (0, negative, very large)
3. **Invalid input**: Non-numeric when number expected, out-of-range values
4. **Type errors**: String when number expected, None values

### Game Scripts
Test scenarios:
1. **Valid moves**: All legal positions/choices work
2. **Invalid moves**: Occupied cells, out-of-range positions rejected
3. **Win conditions**: All winning combinations detected
4. **Draw conditions**: Full board without winner detected
5. **Input validation**: Invalid formats handled gracefully
6. **Menu navigation**: All menu options work correctly

### Data Science Pipelines (Week 5-10)
Validation checklist:
1. **Data loading**: File exists, readable, expected columns present
2. **Missing data**: NaN handling doesn't break pipeline
3. **Edge cases**: Empty dataframe, single row, all same class
4. **Output artifacts**: All expected files created with non-zero size
5. **Reproducibility**: Same random_state produces same results
6. **Optional dependencies**: Pipeline works without optional libraries

## Edge Cases by Type

### Numeric Input
```python
# Test cases to verify
test_values = [
    0,           # Zero
    -1,          # Negative
    1,           # Positive
    999999,      # Large number
    -999999,     # Large negative
    0.5,         # Float (if applicable)
    float('inf') # Infinity (if applicable)
]
```

### String Input
```python
# Test cases to verify
test_strings = [
    "",              # Empty string
    " ",             # Whitespace only
    "a",             # Single character
    "A" * 1000,      # Very long string
    "Test123!@#",    # Mixed characters
    "   test   ",    # Surrounding whitespace
]
```

### List/Collection Input
```python
# Test cases to verify
test_lists = [
    [],              # Empty list
    [1],             # Single element
    [1, 2, 3],       # Normal case
    [1] * 10000,     # Large list
    [None, 1, 2],    # Contains None
    [1, "2", 3.0],   # Mixed types
]
```

### DataFrame Input
```python
# Test scenarios
# 1. Empty DataFrame
df_empty = pd.DataFrame()

# 2. Single row
df_single = pd.DataFrame({'col': [1]})

# 3. Missing values
df_missing = pd.DataFrame({'col': [1, None, 3]})

# 4. All same value
df_constant = pd.DataFrame({'col': [1, 1, 1]})

# 5. Single class (classification)
df_single_class = pd.DataFrame({'target': [0, 0, 0]})
```

## Validation Commands

### Quick Data Checks
```python
# Check data loaded correctly
print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())

# Check value distributions
print(df['column'].value_counts())
print(df.describe())
```

### Pipeline Output Validation
```bash
# Check files were created
ls -lh outputs/
find outputs/ -type f -name "*.csv"
find outputs/ -type f -name "*.png"

# Check file sizes (should be non-zero)
find outputs/ -type f -size 0

# Quick CSV check
head outputs/summary.csv
wc -l outputs/summary.csv

# Check JSON structure
python -m json.tool outputs/results.json
```

### Script Execution Validation
```bash
# Run script and check exit code
python script.py && echo "Success" || echo "Failed"

# Capture output for inspection
python script.py > output.log 2>&1

# Time execution
time python script.py

# Run with different arguments
python script.py --bonus
python script.py --data alternative.csv
```

## Common Test Patterns

### Function Testing Pattern
```python
def validate_email(email):
    """Validate email format."""
    return '@' in email and '.' in email.split('@')[1]

# Manual test cases
test_cases = [
    ("test@example.com", True),
    ("invalid.email", False),
    ("@example.com", False),
    ("test@example", False),
    ("", False),
]

print("Testing validate_email:")
for email, expected in test_cases:
    result = validate_email(email)
    status = "✓" if result == expected else "✗"
    print(f"{status} validate_email({email!r}) = {result} (expected {expected})")
```

### Class Testing Pattern
```python
# Test game class
game = Game()

# Test initial state
assert game.board == [[' ']*3 for _ in range(3)], "Board should be empty"

# Test valid move
game.make_move(0, 0, 'X')
assert game.board[0][0] == 'X', "Move should update board"

# Test invalid move
try:
    game.make_move(0, 0, 'O')  # Cell already occupied
    assert False, "Should raise error for occupied cell"
except ValueError:
    pass  # Expected

print("All game tests passed!")
```

### Pipeline Testing Pattern
```python
def test_pipeline():
    """Run pipeline with test data and validate outputs."""
    # Setup
    test_data_path = "test_data.csv"
    test_output_dir = "test_outputs"
    
    # Run pipeline
    run_pipeline(test_data_path, test_output_dir)
    
    # Validate outputs
    assert (Path(test_output_dir) / "summary.json").exists()
    assert (Path(test_output_dir) / "figures" / "plot.png").exists()
    
    # Check content
    with open(Path(test_output_dir) / "summary.json") as f:
        summary = json.load(f)
        assert "accuracy" in summary
        assert 0 <= summary["accuracy"] <= 1
    
    print("Pipeline test passed!")
```

## Data Quality Validation

### Pre-Pipeline Checks
```python
def validate_dataset(df):
    """Validate dataset before processing."""
    issues = []
    
    # Check not empty
    if df.empty:
        issues.append("Dataset is empty")
    
    # Check for ID columns
    if 'id' in df.columns and df['id'].duplicated().any():
        issues.append("Duplicate IDs found")
    
    # Check missing values
    missing_pct = df.isnull().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        issues.append(f"Columns with >50% missing: {list(high_missing.index)}")
    
    # Check data types
    if df.select_dtypes(include=['object']).columns.any():
        obj_cols = df.select_dtypes(include=['object']).columns
        issues.append(f"Object columns may need encoding: {list(obj_cols)}")
    
    if issues:
        print("⚠ Data quality issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Dataset validation passed")
    
    return len(issues) == 0
```

### Post-Pipeline Checks
```python
def validate_outputs(output_dir):
    """Validate pipeline outputs."""
    output_dir = Path(output_dir)
    
    checks = {
        "Summary exists": (output_dir / "summary.json").exists(),
        "Figures created": len(list((output_dir / "figures").glob("*.png"))) > 0,
        "CSV outputs exist": len(list(output_dir.glob("**/*.csv"))) > 0,
    }
    
    all_passed = all(checks.values())
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all_passed
```

## Interactive Testing

### IPython/Jupyter Testing
```python
# Load intermediate results for inspection
import pandas as pd

# Check data loading
df = pd.read_csv("data.csv")
df.head()
df.info()

# Test transformation
clean_df = df.dropna()
print(f"Dropped {len(df) - len(clean_df)} rows with missing values")

# Visualize distributions
import matplotlib.pyplot as plt
df['column'].hist()
plt.show()

# Test function with different inputs
result1 = process_data(df.head(10))
result2 = process_data(df.tail(10))
```

### Quick Function Testing
```python
# Test in Python REPL or Jupyter
from module import function

# Test with known input
result = function([1, 2, 3, 4, 5])
print(f"Result: {result}")

# Test edge cases
print(function([]))      # Empty
print(function([1]))     # Single element
print(function([1]*100)) # Large input
```

## Error Reproduction

### Isolating Issues
```python
# If pipeline fails at step 3:
# 1. Load data from step 2
df = pd.read_csv("outputs/preprocessing/cleaned_data.csv")

# 2. Run step 3 in isolation
from module import failing_function
result = failing_function(df)

# 3. Inspect intermediate states
print(df.shape)
print(df.dtypes)
print(df.head())
```

### Debugging Workflow
1. **Identify failing stage** from logs or error message
2. **Load last successful state** from outputs
3. **Reproduce error** in isolation
4. **Add debug prints** before failing line
5. **Inspect variables** at failure point
6. **Test fix** with small subset
7. **Verify fix** with full data

## Test Data Creation

### Minimal Test Cases
```python
# Create minimal test data for quick validation
test_df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': ['A', 'B', 'A', 'B', 'A'],
    'target': [0, 1, 0, 1, 0]
})

# Save for repeated testing
test_df.to_csv('test_data.csv', index=False)
```

### Edge Case Test Data
```python
# Create edge case datasets
edge_cases = {
    'empty': pd.DataFrame(columns=['feature1', 'feature2', 'target']),
    'single_row': test_df.head(1),
    'single_class': test_df[test_df['target'] == 0],
    'missing_values': test_df.copy().replace({1: None}),
}

for name, df in edge_cases.items():
    df.to_csv(f'test_{name}.csv', index=False)
```
