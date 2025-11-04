# DI-Bootcamp Coding Guidelines

This is a **Python learning bootcamp repository** organized by weeks and days, covering fundamentals through advanced data science. Code progresses from basic exercises to full-featured mini-projects and ML pipelines.

## Project Architecture

### Directory Structure Pattern
```
WeekX/
  DayY/
    course/          # Lesson examples
    ExerciseXp/      # Practice exercises
    MiniProject/     # Applied projects
    DailyChallenge/  # Individual challenges
```

Advanced projects (Week10+) use **modular pipeline architecture**:
- `config.py` - Constants (paths, settings like `RANDOM_STATE = 42`)
- `pipeline.py` - Orchestration logic that wires all stages
- `*_io.py` or `io_utils.py` - Filesystem and I/O operations
- `data_processing.py` - Cleaning, transformations
- `modeling.py` - ML pipelines, training, evaluation
- `visualization.py` - Chart generation (Matplotlib/Seaborn/Plotly)
- Entry point script delegates to `run_pipeline()` (see `employee_attrition_analysis.py`, `predicting_heart_disease.py`)

### Output Directory Convention
All artifacts write to `outputs/` with categorized subdirs:
- `summaries/` - Descriptive statistics
- `preprocessing/` - Scaled/encoded features
- `statistics/` - Correlation matrices, test results
- `figures/` - PNG visualizations
- `interactive/` - Plotly HTML dashboards
- `reports/` - Narrative summaries
- `logs/` - Execution logs

## Code Patterns

### Python Fundamentals (Weeks 1-4)
- **Functions over inline logic** - Game loops extract validation/display to named functions (see `mini_project_tic_tac_toe.py`, `mini_project_hangman.py`)
- **Input validation loops** - `while True` until valid, with error messages
- **Docstrings everywhere** - Google-style parameter descriptions even in exercises

### OOP Patterns (Weeks 2-4)
- **Class-based game engines** - `Game` class encapsulates logic (see `rock_paper_scissors/game.py`)
- **Class variables for shared state** - `CHOICES`, `WINNING_COMBINATIONS` at class level
- **Inheritance for specialization** - `BlockedDoor(Door)` overrides parent methods (see `intro_oop.py`)
- **Separation of concerns** - Game logic in `game.py`, UI/menu in `rock-paper-scissors.py`

### Data Science Pipelines (Weeks 5-10)
- **Graceful optional dependencies**:
  ```python
  try:
      import seaborn as sns
      HAS_SEABORN = True
  except ImportError:
      HAS_SEABORN = False
  ```
  Check flags before feature use, log warnings at pipeline end
  
- **Structured logging over print** - Use `logger.info()` with context:
  ```python
  logger.info("Data prepared", extra={'rows': df.shape[0], 'cols': df.shape[1]})
  ```

- **Stratified splits for imbalanced data**:
  ```python
  train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
  ```

- **GridSearchCV + StratifiedKFold** - Standard tuning pattern (see `modeling.py`)

- **Bonus/optional features via CLI flags** - `--bonus` for extended model comparisons

### Flask Web Apps (Week 3)
- **Explicit template paths** - `Flask(__name__, template_folder=os.path.join(basedir, 'templates'))`
- **Static JS/CSS** - Frontend interacts via `/api/*` routes returning JSON
- **Error handling in routes** - Try/except with `jsonify({'success': False, 'error': str(e)})`, 500

### API Clients (Week 3)
- **OO wrappers around REST APIs** - `ChuckNorrisClient` class encapsulates endpoints
- **Timing metrics** - `perf_counter()` around requests, return `(data, elapsed_ms)`
- **Type hints with `from __future__ import annotations`** - Enables forward references

## Coding Style & Conventions

### Naming Conventions
- **snake_case everywhere** - Functions, variables, files: `load_and_prepare()`, `numeric_cols`, `data_processing.py`
- **UPPER_CASE for constants** - Module-level constants: `RANDOM_STATE`, `CHOICES`, `WINNING_COMBINATIONS`
- **PascalCase for classes** - `Game`, `AnagramChecker`, `ChuckNorrisClient`, `PreprocessingArtifacts`
- **Descriptive names** - Prefer `clean_df` over `df2`, `class_balance_path` over `path1`

### Documentation Style
- **Docstrings on everything** - Even simple functions in exercises get Google-style docstrings:
  ```python
  def validate_input(user_input):
      """
      Validate the user's input and return cleaned input or None if invalid.
      
      Args:
          user_input (str): The raw input string from the user
      
      Returns:
          str or None: The cleaned input string if valid, None if invalid
      """
  ```
- **Inline comments for clarity** - Explain **why**, not **what** (see numpy operations in `Week7/`)
- **File-level docstrings** - Advanced modules start with purpose/usage (see `chuck_norris_api.py`)

### Code Organization
- **One concern per function** - Game loops extract validation, display, checking to separate functions
- **Type hints in advanced code** - Week10+ uses `from __future__ import annotations` + type hints
- **Structured logging over print** - Pipelines use `logger.info("Step complete", extra={...})` instead of print
- **Progress indicators** - Pipeline scripts use custom `_log()` with structured `[FLOW]` tags for tracking

### Import Order
1. Future imports: `from __future__ import annotations`
2. Standard library: `import os`, `from pathlib import Path`
3. Third-party: `import pandas as pd`, `import numpy as np`
4. Local modules: `from config import`, `from pipeline import`

### Best Practices
- **Early validation** - Check file existence before loading (raise `FileNotFoundError` with path)
- **No magic numbers** - Extract to constants in `config.py` (paths, random state, thresholds)
- **Fail fast** - Validate inputs at function entry, raise exceptions early
- **Return tuples for multi-values** - `return (data, elapsed_ms)` instead of dict for simple pairs
- **Entry point pattern** - Always include `if __name__ == "__main__":` for runnable scripts

## Running Projects

### Simple Scripts
```bash
python path/to/script.py
```

### Pipelines with Arguments
```bash
# Data science projects
python predicting_heart_disease.py --data dataset.csv --out outputs --bonus

# Superstore analysis
python superstore_marketing_analysis.py --file "US Superstore data.xls"
```

### Flask Apps
```bash
cd Week3/Day5/course_api/chuck_norris
python app.py
# Then open browser to http://localhost:5000
```

### Dependencies
Install from root:
```bash
pip install -r requirements.txt
```
Core stack: pandas, numpy, matplotlib, scikit-learn, Flask. Optional: seaborn, scipy, plotly, plotnine, xgboost.

## Testing & Debugging

### Validation Approach
- **No unit tests** - This is a learning repo; validation happens through execution
- **Manual verification** - Run scripts and inspect artifacts in `outputs/`
- **Check logs** - Advanced projects log to both console and `outputs/logs/pipeline.log`

### Debugging Workflow
1. **Read logs first** - Pipeline logs contain structured progress messages with `[FLOW]` or `[levelname]` tags
2. **Check output artifacts** - Missing/incorrect CSV/PNG/JSON files indicate where pipeline failed
3. **Reproduce in isolation** - Extract failing module (e.g., `data_processing.py`) and test standalone
4. **Interactive testing** - For data issues, load intermediate CSVs in Jupyter/IPython for inspection

### Error Handling Patterns
- **Try/except for optional deps** - Wrap imports in try/except, set `HAS_LIBRARY` flags:
  ```python
  try:
      import seaborn as sns
      HAS_SEABORN = True
  except ImportError:
      HAS_SEABORN = False
  ```
- **Validation loops** - User input repeats until valid (see `mini_project_tic_tac_toe.py`, `anagram_checker/anagrams.py`)
- **Specific exceptions** - Raise `ValueError`, `TypeError`, `FileNotFoundError` with descriptive messages
- **Flask route errors** - Return `jsonify({'success': False, 'error': str(e)})` with 500 status

### Troubleshooting Tips
- **Matplotlib display issues** - All plots save to files; no interactive backend needed (headless-safe)
- **Missing output directories** - Pipelines auto-create via `Path.mkdir(parents=True, exist_ok=True)`
- **Import errors in Week10** - Advanced projects use bare imports, not relative (e.g., `from pipeline import`, not `from . import`)
- **DataFrame encoding issues** - Files use UTF-8; logging configured with `encoding="utf-8"`

## Common Pitfalls

1. **Excel file formats** - `.xls` requires `xlrd`, `.xlsx` works with `openpyxl` (both in requirements)
2. **Relative imports** - Subpackages use bare imports (`from pipeline import`), not relative (`from . import`)
3. **RANDOM_STATE consistency** - Always use `RANDOM_STATE = 42` from config for reproducibility
4. **Warnings suppression** - `warnings.filterwarnings("ignore")` at pipeline start to reduce noise

## Extending Projects

- **Add visualization** - Follow pattern in `visualization.py`: separate function per chart type, save to `FIGURES_DIR`
- **New features** - Add module in project folder, import in `pipeline.py`, call in orchestration
- **CLI arguments** - Use `argparse` with defaults (see `predicting_heart_disease.py` for examples)

## Key Files to Reference

- Pipeline architecture: `Week10/Day5/hackaton/employee_attri_performance/`
- OOP games: `Week4/Day1/mini_project/rock_paper_scissors/`
- Flask API: `Week3/Day5/course_api/chuck_norris/`
- ML workflow: `Week10/Day3/miniproject/predicting_heart_disease.py`
