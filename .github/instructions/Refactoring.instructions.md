---
applyTo: '**/*.py'
---

# Refactoring Guidelines

## Progressive Complexity Principles

### Week 1-2 Style (Beginner)
- **Inline logic**: Everything in main script
- **Simple functions**: `def calculate(a, b): return a + b`
- **Basic variables**: `result`, `total`, `count`
- **Print statements**: Direct `print()` calls
- **Simple error handling**: `try/except` when necessary
- **Comments**: Exercise instructions preserved in comments

### Week 3-4 Style (Intermediate)
- **Function extraction**: Logic grouped into named functions
- **Class introduction**: When multiple related functions exist
- **Docstrings**: Google-style documentation added
- **Input validation**: Dedicated validation functions
- **Module separation**: Logic vs UI split into files
- **Error messages**: User-friendly validation messages

### Week 5-7 Style (Advanced Data)
- **Pandas/NumPy**: Vectorized operations over loops
- **Function composition**: Chain operations for clarity
- **Configuration**: Constants extracted to top of file
- **Structured outputs**: Save to organized directories
- **Comments**: Focus on "why" not "what"

### Week 10 Style (Expert Pipeline)
- **Modular architecture**: `config.py`, `pipeline.py`, specialized modules
- **Type hints**: `from __future__ import annotations`
- **Logging**: Structured logging instead of print
- **Configuration module**: All paths and constants centralized
- **Orchestration**: `pipeline.py` coordinates all stages
- **Graceful degradation**: Optional dependencies with flags

## When to Extract Functions

### Extract When:
1. **Code block repeats** 2+ times
2. **Logical unit** performs single task
3. **Improves readability** by naming the operation
4. **Testable independently** from rest of code
5. **Exceeds 20-30 lines** in a single block

### Example Progression

**Week 1 - Inline:**
```python
user_input = input("Enter a number: ")
while not user_input.isdigit():
    print("Invalid input. Please enter a number.")
    user_input = input("Enter a number: ")
number = int(user_input)
```

**Week 3 - Extracted:**
```python
def get_valid_number(prompt="Enter a number: "):
    """Get a valid integer from user input."""
    while True:
        user_input = input(prompt)
        if user_input.isdigit():
            return int(user_input)
        print("Invalid input. Please enter a number.")

number = get_valid_number()
```

## When to Introduce Classes

### Stay with Functions When:
- Simple script with few operations
- No shared state between operations
- Linear processing flow
- Early-week exercises (Week 1-2)

### Introduce Classes When:
- **Multiple related functions** share data
- **State management** needed (game state, configuration)
- **Encapsulation** improves code organization
- **Multiple instances** may exist
- Week 3+ complexity level

### Example Progression

**Functions (Week 2):**
```python
def check_win(board, player):
    # Check logic
    pass

def make_move(board, position, player):
    # Move logic
    pass

# Use functions
board = [[' ']*3 for _ in range(3)]
make_move(board, (0, 0), 'X')
```

**Class (Week 4):**
```python
class Game:
    """Encapsulates game logic and state."""
    
    WINNING_COMBINATIONS = {...}
    
    def __init__(self):
        self.board = [[' ']*3 for _ in range(3)]
    
    def check_win(self, player):
        # Check using self.board
        pass
    
    def make_move(self, position, player):
        # Update self.board
        pass

# Use class
game = Game()
game.make_move((0, 0), 'X')
```

## When to Add Type Hints

### Without Type Hints (Weeks 1-7):
```python
def calculate_total(prices, tax_rate):
    """Calculate total with tax."""
    return sum(prices) * (1 + tax_rate)
```

### With Type Hints (Week 10+):
```python
from __future__ import annotations

def calculate_total(prices: list[float], tax_rate: float) -> float:
    """Calculate total with tax."""
    return sum(prices) * (1 + tax_rate)
```

### When to Add:
- Week 10+ projects
- Complex function signatures
- Public API functions
- When using dataclasses or Pydantic
- After functionality is working

### When to Skip:
- Early learning exercises
- Obvious parameter types
- One-off scripts
- Rapid prototyping

## When to Switch from Print to Logging

### Use Print For:
- Week 1-7 exercises
- Simple scripts
- Direct user interaction
- Game messages and displays
- Quick debugging

### Use Logging For:
- Week 10 pipelines
- Long-running processes
- Background tasks
- Production-like code
- When you need log files

### Migration Pattern

**Before (print):**
```python
print(f"Processing {len(data)} rows...")
# process
print("Processing complete!")
```

**After (logging):**
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processing %d rows", len(data))
# process
logger.info("Processing complete")
```

## Function Extraction Patterns

### Extract Validation
**Before:**
```python
user_input = input("Enter row: ")
while not user_input.isdigit() or not 1 <= int(user_input) <= 3:
    print("Invalid input")
    user_input = input("Enter row: ")
row = int(user_input) - 1
```

**After:**
```python
def get_valid_row():
    """Get valid row number (1-3) from user."""
    while True:
        user_input = input("Enter row: ")
        try:
            row = int(user_input)
            if 1 <= row <= 3:
                return row - 1  # Convert to 0-indexed
            print("Row must be 1, 2, or 3.")
        except ValueError:
            print("Please enter a number.")
```

### Extract Display Logic
**Before:**
```python
print("="*50)
print(f"Score: {score}")
print(f"Attempts: {attempts}")
print("="*50)
```

**After:**
```python
def display_status(score: int, attempts: int) -> None:
    """Display current game status."""
    print("="*50)
    print(f"Score: {score}")
    print(f"Attempts: {attempts}")
    print("="*50)
```

### Extract Computation
**Before:**
```python
total = sum(values)
average = total / len(values)
variance = sum((x - average)**2 for x in values) / len(values)
std_dev = variance ** 0.5
```

**After:**
```python
def calculate_statistics(values: list[float]) -> dict[str, float]:
    """Calculate descriptive statistics for values."""
    total = sum(values)
    n = len(values)
    average = total / n
    variance = sum((x - average)**2 for x in values) / n
    
    return {
        'mean': average,
        'variance': variance,
        'std_dev': variance ** 0.5
    }
```

## Configuration Extraction

### Week 1-5: Constants at Top
```python
# Constants
MAX_ATTEMPTS = 6
BOARD_SIZE = 3
WINNING_SCORE = 100

# Rest of code
```

### Week 10: Dedicated config.py
```python
# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
```

```python
# main.py
from config import RANDOM_STATE, TEST_SIZE, OUTPUT_DIR

# Use configuration
train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
```

## Docstring Evolution

### Week 1-2: Optional
```python
def add(a, b):
    return a + b
```

### Week 3-4: Basic Docstrings
```python
def add(a, b):
    """Add two numbers and return the result."""
    return a + b
```

### Week 5+: Google Style
```python
def add(a, b):
    """
    Add two numbers and return the result.
    
    Args:
        a (int|float): First number
        b (int|float): Second number
    
    Returns:
        int|float: Sum of a and b
    """
    return a + b
```

### Week 10: With Type Hints
```python
def add(a: float, b: float) -> float:
    """
    Add two numbers and return the result.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Sum of a and b
    """
    return a + b
```

## Refactoring Checklist

When improving existing code:

1. ✅ **Functionality first**: Ensure code works before refactoring
2. ✅ **One change at a time**: Extract functions, then add classes, then add types
3. ✅ **Test after each change**: Verify code still works
4. ✅ **Preserve behavior**: Don't change what the code does
5. ✅ **Match week complexity**: Don't over-engineer early exercises
6. ✅ **Add documentation**: Update docstrings when changing signatures
7. ✅ **Consider readability**: Simpler is often better than clever
