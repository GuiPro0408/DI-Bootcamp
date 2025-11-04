---
applyTo: '**/*.py'
---

# Error Message Guidelines

## User-Facing Error Principles

### Be Specific and Actionable
**Bad:**
```python
if not valid:
    print("Error")
```

**Good:**
```python
if not email_valid:
    print("Error: Email must contain '@' and a domain (e.g., user@example.com)")
```

### Provide Context
**Bad:**
```python
raise ValueError("Invalid input")
```

**Good:**
```python
raise ValueError(f"Invalid age: {age}. Age must be between 0 and 120.")
```

### Suggest Solutions
**Bad:**
```python
print("File not found")
```

**Good:**
```python
print(f"Error: File '{filename}' not found.")
print(f"Please check that the file exists in: {os.getcwd()}")
print("Or provide the full path to the file.")
```

## Validation Error Patterns

### Input Format Errors
```python
def get_valid_number(prompt="Enter a number: "):
    """Get a valid integer from user with clear error messages."""
    while True:
        user_input = input(prompt)
        
        # Check not empty
        if not user_input.strip():
            print("Error: Input cannot be empty. Please enter a number.")
            continue
        
        # Check is digit
        if not user_input.isdigit():
            print(f"Error: '{user_input}' is not a valid number.")
            print("Please enter digits only (0-9).")
            continue
        
        return int(user_input)
```

### Range Validation Errors
```python
def get_board_position():
    """Get valid board position (1-3) with helpful errors."""
    while True:
        try:
            position = int(input("Enter position (1-3): "))
            
            if not 1 <= position <= 3:
                print(f"Error: Position must be between 1 and 3, not {position}.")
                print("Available positions: 1, 2, 3")
                continue
            
            return position - 1  # Convert to 0-indexed
            
        except ValueError as e:
            print(f"Error: Please enter a number, not text.")
```

### State Validation Errors
```python
def make_move(board, row, col):
    """Make a move with state validation."""
    # Check position is on board
    if not (0 <= row < 3 and 0 <= col < 3):
        raise ValueError(
            f"Position ({row}, {col}) is outside board boundaries. "
            f"Valid range: 0-2 for both row and column."
        )
    
    # Check position is empty
    if board[row][col] != ' ':
        raise ValueError(
            f"Position ({row}, {col}) is already occupied by '{board[row][col]}'. "
            f"Please choose an empty cell."
        )
    
    # Make move
    board[row][col] = 'X'
```

## File Operation Errors

### File Not Found
```python
from pathlib import Path

def load_data(filepath):
    """Load data with helpful error message."""
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Searched in: {path.absolute()}\n"
            f"Current directory: {Path.cwd()}\n"
            f"Please check:\n"
            f"  1. File name is spelled correctly\n"
            f"  2. File is in the expected directory\n"
            f"  3. You have read permissions"
        )
    
    return pd.read_csv(path)
```

### Permission Errors
```python
def save_results(data, output_path):
    """Save with permission error handling."""
    try:
        data.to_csv(output_path)
    except PermissionError:
        print(f"Error: Cannot write to {output_path}")
        print("Possible reasons:")
        print("  - File is open in another program")
        print("  - You don't have write permission")
        print("  - Directory is read-only")
        print(f"\nTry saving to a different location or closing the file.")
        raise
```

### Import Errors (Optional Dependencies)
```python
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  Warning: seaborn not installed")
    print("Some visualizations will use basic matplotlib styling.")
    print("To install: pip install seaborn")
```

## Progressive Hint System

### Game Input Validation
```python
def get_valid_guess(guessed_letters):
    """Get valid letter guess with progressive hints."""
    while True:
        guess = input("Guess a letter: ").lower().strip()
        
        # Hint 1: Basic format
        if len(guess) != 1:
            print("❌ Please enter exactly one letter.")
            continue
        
        # Hint 2: Must be letter
        if not guess.isalpha():
            print("❌ Please enter a letter (a-z), not a number or symbol.")
            continue
        
        # Hint 3: Already guessed
        if guess in guessed_letters:
            print(f"⚠️  You already guessed '{guess}'.")
            print(f"Already guessed: {', '.join(sorted(guessed_letters))}")
            continue
        
        return guess
```

### Calculation Validation
```python
def divide_numbers(a, b):
    """Divide with clear error handling."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print(f"❌ Error: Cannot divide {a} by zero.")
        print("Division by zero is mathematically undefined.")
        print("Please use a non-zero divisor.")
        raise
    except TypeError as e:
        print(f"❌ Error: Invalid types for division.")
        print(f"Got: {type(a).__name__} / {type(b).__name__}")
        print("Both values must be numbers (int or float).")
        raise
```

## When to Raise vs Return None

### Raise Exceptions For:
- **Programming errors**: Wrong types, invalid state
- **Unrecoverable errors**: File not found, network error
- **Constraint violations**: Out of range, invalid format

```python
def calculate_age(birth_year):
    """Calculate age from birth year."""
    current_year = 2025
    
    if not isinstance(birth_year, int):
        raise TypeError(f"Birth year must be int, not {type(birth_year).__name__}")
    
    if birth_year > current_year:
        raise ValueError(f"Birth year {birth_year} cannot be in the future")
    
    if birth_year < 1900:
        raise ValueError(f"Birth year {birth_year} seems unrealistic")
    
    return current_year - birth_year
```

### Return None For:
- **Optional results**: Search with no matches
- **Validation failures**: User input retry loop
- **Graceful degradation**: Optional features

```python
def find_player(players, name):
    """Find player by name, return None if not found."""
    for player in players:
        if player['name'] == name:
            return player
    return None  # Not found - caller handles it

# Usage
player = find_player(players, "Alice")
if player is None:
    print("Player not found. Please try again.")
else:
    print(f"Found: {player['name']}")
```

## Error Message Formatting

### Console Error Format
```python
def format_error(title, message, suggestions=None):
    """Format error message with consistent style."""
    lines = [
        "=" * 60,
        f"❌ ERROR: {title}",
        "=" * 60,
        message,
    ]
    
    if suggestions:
        lines.append("\n💡 Suggestions:")
        for suggestion in suggestions:
            lines.append(f"  • {suggestion}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

# Usage
error_msg = format_error(
    title="Invalid Configuration",
    message="Required setting 'API_KEY' not found in config.",
    suggestions=[
        "Check config.py exists",
        "Verify API_KEY is defined",
        "Try setting API_KEY='your-key-here'"
    ]
)
print(error_msg)
```

### Data Validation Errors
```python
def validate_dataframe(df):
    """Validate dataframe with detailed error messages."""
    errors = []
    
    if df.empty:
        errors.append("Dataset is empty (0 rows)")
    
    if 'id' in df.columns and df['id'].duplicated().any():
        dup_count = df['id'].duplicated().sum()
        errors.append(f"Found {dup_count} duplicate IDs")
    
    missing = df.isnull().sum()
    high_missing = missing[missing > len(df) * 0.5]
    if not high_missing.empty:
        cols = ', '.join(high_missing.index)
        errors.append(f"Columns with >50% missing: {cols}")
    
    if errors:
        error_msg = "\n".join([
            "Data Validation Failed:",
            *[f"  ❌ {e}" for e in errors],
            "\nPlease fix these issues before proceeding."
        ])
        raise ValueError(error_msg)
```

## Logging vs Printing Errors

### Print for User-Facing Errors (Weeks 1-7)
```python
# Simple scripts, games, exercises
if invalid_input:
    print("Error: Invalid input. Please try again.")
```

### Log for Pipeline Errors (Week 10)
```python
# Advanced pipelines
try:
    df = load_data(path)
    logger.info("Data loaded successfully: %d rows", len(df))
except FileNotFoundError as e:
    logger.error("Failed to load data: %s", e)
    logger.error("Attempted path: %s", path)
    raise
```

### Both for Critical Errors
```python
# Important errors that need user attention
try:
    result = critical_operation()
except Exception as e:
    logger.error("Critical operation failed: %s", e, exc_info=True)
    print("\n❌ CRITICAL ERROR")
    print(f"Operation failed: {e}")
    print("Check logs for details.")
    raise
```

## Error Message Testing

### Test Edge Cases
```python
# Test validation with edge cases
test_cases = [
    ("", "Empty input"),
    ("   ", "Whitespace only"),
    ("abc", "Non-numeric"),
    ("-5", "Negative"),
    ("0", "Zero"),
    ("999999", "Very large"),
]

for input_value, description in test_cases:
    print(f"\nTesting: {description}")
    try:
        result = validate_input(input_value)
        print(f"✓ Accepted: {result}")
    except ValueError as e:
        print(f"✗ Rejected: {e}")
```

## Error Recovery Patterns

### Retry Loop
```python
def get_valid_choice(options):
    """Get valid choice with unlimited retries."""
    attempts = 0
    max_hints = 3
    
    while True:
        attempts += 1
        choice = input(f"Choose from {options}: ").strip()
        
        if choice in options:
            return choice
        
        # Progressive help
        print(f"❌ '{choice}' is not a valid choice.")
        
        if attempts <= max_hints:
            print(f"Valid options: {', '.join(options)}")
        
        if attempts == max_hints:
            print(f"💡 Hint: Type exactly one of these: {' or '.join(options)}")
```

### Fallback Options
```python
def load_config(primary_path, fallback_path=None):
    """Load config with fallback."""
    try:
        return load_from_file(primary_path)
    except FileNotFoundError:
        print(f"⚠️  Primary config not found: {primary_path}")
        
        if fallback_path:
            print(f"Trying fallback: {fallback_path}")
            try:
                return load_from_file(fallback_path)
            except FileNotFoundError:
                print(f"❌ Fallback also not found: {fallback_path}")
                raise
        else:
            print("No fallback configured. Using defaults.")
            return get_default_config()
```
