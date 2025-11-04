---
applyTo: '**/*.{py,md}'
---

# Documentation Standards

## When to Write README vs Inline Comments

### README Files
Write README.md when:
- **Project/directory** has multiple related files
- **Week10 mini-projects** with complex workflows
- **API or library** that others will use
- **Setup instructions** are non-obvious
- **Usage examples** would help understanding

README should include:
1. **Purpose**: What the project does
2. **Structure**: File organization and responsibilities
3. **Usage**: How to run with examples
4. **Dependencies**: Required packages
5. **Outputs**: What artifacts are generated

### Inline Comments
Use inline comments for:
- **Non-obvious logic**: Why not what
- **Algorithm explanations**: Key steps in complex logic
- **Edge case handling**: Why special cases exist
- **Temporary workarounds**: Mark with TODO/FIXME
- **Exercise instructions**: Preserve problem statements in Weeks 1-4

Don't comment:
- Obvious operations: `x = 5  # set x to 5`
- Self-explanatory code: Well-named functions/variables
- Repeated patterns: Comment first occurrence only

## Docstring Detail by Week

### Week 1-2: Optional/Minimal
```python
def calculate_total(items):
    """Calculate the total price of items."""
    return sum(items)
```

### Week 3-4: Single Line with Context
```python
def validate_email(email):
    """
    Validate email format and return True if valid, False otherwise.
    """
    return '@' in email and '.' in email.split('@')[1]
```

### Week 5-7: Google Style with Args/Returns
```python
def calculate_statistics(values):
    """
    Calculate descriptive statistics for a list of values.
    
    Args:
        values (list[float]): Numeric values to analyze
    
    Returns:
        dict: Dictionary containing mean, median, std_dev
    """
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std_dev': np.std(values)
    }
```

### Week 10: Comprehensive with Type Hints
```python
from __future__ import annotations

def preprocess_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler, OneHotEncoder]:
    """
    Preprocess features by scaling numeric and encoding categorical columns.
    
    This function applies StandardScaler to numeric features and OneHotEncoder
    to categorical features, returning both the transformed dataframe and
    fitted transformers for later use on test data.
    
    Args:
        df: Input dataframe with features to preprocess
        numeric_cols: List of numeric column names to scale
        categorical_cols: List of categorical column names to encode
    
    Returns:
        Tuple containing:
        - Preprocessed dataframe with scaled and encoded features
        - Fitted StandardScaler instance
        - Fitted OneHotEncoder instance
    
    Raises:
        ValueError: If specified columns are not found in dataframe
    """
    # Implementation
```

## Example Usage in Docstrings

### Simple Example
```python
def format_currency(amount):
    """
    Format a numeric amount as currency string.
    
    Example:
        >>> format_currency(1234.56)
        '$1,234.56'
    """
    return f"${amount:,.2f}"
```

### Complex Example
```python
def train_model(X_train, y_train, params=None):
    """
    Train a classification model with optional hyperparameters.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        params (dict, optional): Hyperparameters for the model
    
    Returns:
        Trained model instance
    
    Example:
        >>> X_train = np.array([[1, 2], [3, 4]])
        >>> y_train = np.array([0, 1])
        >>> model = train_model(X_train, y_train)
        >>> predictions = model.predict(X_train)
    """
    # Implementation
```

### Pipeline Example
```python
def run_pipeline(data_path: str, output_dir: str) -> None:
    """
    Execute the complete data analysis pipeline.
    
    The pipeline performs these steps:
    1. Load and clean data
    2. Generate descriptive statistics
    3. Create visualizations
    4. Train and evaluate models
    5. Generate reports
    
    Args:
        data_path: Path to input CSV file
        output_dir: Directory for output artifacts
    
    Usage:
        >>> run_pipeline('data.csv', 'outputs/')
        
    The pipeline creates the following structure:
        outputs/
        ├── summaries/
        ├── figures/
        ├── reports/
        └── logs/
    """
    # Implementation
```

## Pipeline Stage Documentation

### Module-Level Docstring
```python
"""
Data preprocessing module for the employee attrition pipeline.

This module handles data cleaning, feature engineering, scaling, and encoding.
It produces preprocessed datasets ready for modeling and statistical analysis.

Functions:
    clean_data: Remove invalid rows and handle missing values
    preprocess_features: Scale numeric and encode categorical features
    summarize_dataset: Generate descriptive statistics
"""

# Imports and code
```

### Function Documentation for Pipeline Stages
```python
def clean_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Clean the raw dataset by handling missing values and invalid entries.
    
    Cleaning steps:
    1. Remove duplicate rows
    2. Drop columns with >50% missing values
    3. Impute remaining missing values (median for numeric, mode for categorical)
    4. Remove outliers beyond 3 standard deviations
    
    Args:
        df: Raw input dataframe
        logger: Logger instance for tracking progress
    
    Returns:
        Cleaned dataframe ready for analysis
    
    Side Effects:
        Logs cleaning statistics (rows dropped, columns removed, etc.)
    """
    # Implementation
```

## Mathematical Notation in Comments

### Statistical Formulas
```python
# Calculate Pearson correlation coefficient
# r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)
correlation = np.corrcoef(x, y)[0, 1]

# Compute z-score normalization
# z = (x - μ) / σ
z_scores = (values - values.mean()) / values.std()
```

### Machine Learning Formulas
```python
# Cross-entropy loss for binary classification
# L = -[y log(ŷ) + (1-y) log(1-ŷ)]
loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# F1 Score: Harmonic mean of precision and recall
# F1 = 2 × (precision × recall) / (precision + recall)
f1 = 2 * (precision * recall) / (precision + recall)
```

### Vectorized Operations
```python
# Apply softmax activation: σ(z_i) = e^(z_i) / Σ(e^(z_j))
exp_scores = np.exp(scores)
probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Calculate weighted sum: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
output = np.dot(weights, features) + bias
```

## README Structure for Projects

### Week 6-7 Mini-Projects
```markdown
# Project Title

Brief description of what the project does.

## Files
- `main.py` - Main analysis script
- `data.csv` - Input dataset
- `requirements.txt` - Dependencies

## How to Run
```bash
python main.py --file data.csv
```

## Outputs
Results are saved to `outputs/`:
- CSV summaries
- PNG visualizations
- Analysis reports
```

### Week 10 Advanced Projects
```markdown
# Project Title

Comprehensive description of the analysis or model.

## Prerequisites
- Python 3.10+
- Required packages: pandas, numpy, matplotlib, scikit-learn
- Optional: seaborn, plotly, xgboost

## Project Structure
```
project/
├── config.py           # Constants and paths
├── pipeline.py         # Orchestration logic
├── data_processing.py  # Cleaning and transformations
├── modeling.py         # ML pipelines
├── visualization.py    # Chart generation
└── main.py            # CLI entry point
```

## Usage

### Basic Usage
```bash
python main.py
```

### With Options
```bash
python main.py --data custom_data.csv --out results/ --bonus
```

### Arguments
- `--data`: Path to input CSV (default: bundled dataset)
- `--out`: Output directory (default: `outputs/`)
- `--bonus`: Run extended model comparison

## Pipeline Stages
1. **Data Loading**: Load and validate input
2. **Cleaning**: Handle missing values and outliers
3. **EDA**: Generate descriptive statistics and plots
4. **Preprocessing**: Scale and encode features
5. **Modeling**: Train and evaluate models
6. **Reporting**: Generate insights and recommendations

## Output Structure
```
outputs/
├── summaries/       # Descriptive statistics
├── preprocessing/   # Scaled/encoded features
├── statistics/      # Correlation matrices
├── figures/         # PNG visualizations
├── reports/         # Narrative summaries
└── logs/           # Execution logs
```

## Troubleshooting
- **Import errors**: Install requirements with `pip install -r requirements.txt`
- **File not found**: Check data path is correct
- **Missing plots**: Optional dependencies required (seaborn, plotly)
```

## API/Library Documentation

### Module Documentation
```python
"""
Chuck Norris API Client

OO wrapper around the public Chuck Norris jokes API.

Classes:
    ChuckNorrisClient: Main API client with timing metrics

Usage:
    >>> from chuck_norris_api import ChuckNorrisClient
    >>> client = ChuckNorrisClient()
    >>> joke, joke_id, elapsed = client.random_joke()
    >>> print(joke)
"""
```

### Class Documentation
```python
class ChuckNorrisClient:
    """
    Client for interacting with the Chuck Norris jokes API.
    
    This client provides methods to fetch random jokes, search jokes,
    and retrieve available categories. All methods return timing information
    for performance monitoring.
    
    Attributes:
        BASE_URL (str): Base URL for the API
        session (requests.Session): Persistent session for connections
    
    Example:
        >>> client = ChuckNorrisClient()
        >>> joke, joke_id, elapsed = client.random_joke()
        >>> print(f"Joke fetched in {elapsed:.2f}ms")
    """
```

## Comment Style Guidelines

### Explain Why, Not What
**Bad:**
```python
# Set x to 5
x = 5

# Loop through items
for item in items:
    # Add 1 to item
    result = item + 1
```

**Good:**
```python
# Use 5 as default timeout to balance responsiveness and reliability
x = 5

# Transform raw values to normalized scale
for item in items:
    result = item + 1
```

### Section Separators
```python
# ===========================
# Data Loading
# ===========================

# ===========================
# Feature Engineering
# ===========================

# ===========================
# Model Training
# ===========================
```

### TODO/FIXME/NOTE Markers
```python
# TODO: Add support for categorical features
# FIXME: Memory leak when processing large datasets
# NOTE: This approach works but is slow for large N
# HACK: Temporary workaround until API v2 is available
# XXX: This code is dangerous and needs review
```

### Complex Logic Explanation
```python
# Check if the player has achieved any winning combination
# We compare each possible win state (row, column, diagonal)
# against a list of the player's symbols
# e.g., ['X', 'X', 'X'] in win_states
return [player] * 3 in win_states
```

## Documentation Checklist

Before committing:
- [ ] All public functions have docstrings
- [ ] Complex algorithms are explained with comments
- [ ] README exists for multi-file projects
- [ ] Usage examples are provided for non-obvious code
- [ ] Type hints added for Week 10+ code
- [ ] Mathematical formulas are documented
- [ ] Edge cases and assumptions are noted
- [ ] TODOs removed or explained
