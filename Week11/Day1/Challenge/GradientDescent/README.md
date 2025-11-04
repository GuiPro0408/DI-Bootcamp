# Challenge: Predicting Student Exam Scores Using Gradient Descent

## Overview

This challenge implements **one step of gradient descent** for a simple neural network that predicts a student's exam score based on:
- **Study Hours (x₁)**: Number of hours studied for the exam
- **Previous Test Score (x₂)**: Score from the last test

The goal is to demonstrate how neural networks learn by adjusting their parameters to reduce prediction error.

## Problem Setup

### Initial Parameters
- **w₁ = 0.6** (weight for study hours)
- **w₂ = 0.3** (weight for previous test score)
- **b = 10** (bias term)
- **α = 0.01** (learning rate)

### Sample Data
- **x₁ = 5** (study hours)
- **x₂ = 70** (previous test score)
- **y = 85** (actual exam score)

## Running the Challenge

```bash
python gradient_descent_challenge.py
```

The script will:
1. Perform a complete gradient descent step
2. Display progress messages to the console
3. Generate a detailed report in `outputs/gradient_descent_report.txt`

## What You'll Learn

The implementation demonstrates:

### 1. Forward Pass
Calculate prediction using the formula:
```
ŷ = w₁·x₁ + w₂·x₂ + b
```

### 2. Loss Calculation
Measure error using Mean Squared Error:
```
L = (1/2)(y - ŷ)²
```

### 3. Gradient Computation
Calculate partial derivatives:
```
∂L/∂w₁ = -(y - ŷ)·x₁
∂L/∂w₂ = -(y - ŷ)·x₂
∂L/∂b = -(y - ŷ)
```

### 4. Parameter Update
Apply gradient descent rule:
```
θ_new = θ_old - α·∂L/∂θ
```

## Output

The script generates `outputs/gradient_descent_report.txt` with:
- Step-by-step calculations with intermediate values
- Mathematical formulas and explanations
- Before/after comparison showing parameter changes
- Educational observations about the learning process

## Key Observations

This implementation intentionally shows a case where the prediction gets **worse** after one step, demonstrating:
- Why **feature scaling** is crucial (x₂ = 70 is much larger than x₁ = 5)
- The importance of **learning rate tuning**
- Why **multiple iterations** are needed for convergence
- The necessity of **proper data preprocessing** in real machine learning

In practice, neural networks:
- Normalize input features to similar scales (0-1 range or standardized)
- Process many training examples (batch/mini-batch gradient descent)
- Run for many iterations (epochs) until loss converges
- Use advanced techniques like momentum or adaptive learning rates

## Code Structure

The implementation follows DI-Bootcamp Week 5-7 conventions:
- **Google-style docstrings** with parameter descriptions
- **Structured logging** with `[FLOW]` tags for progress tracking
- **Snake_case naming** for functions and variables
- **UPPER_CASE constants** for configuration values
- **Type hints** for function signatures
- **Comprehensive comments** explaining mathematical concepts
- **File output** following Week 10 patterns

## Mathematical Background

This challenge demonstrates the foundation of how neural networks learn:
1. Make a prediction with current parameters
2. Calculate how wrong the prediction is (loss)
3. Compute gradients showing how to improve
4. Update parameters to reduce error
5. Repeat until the model converges

This single-step example is simplified for educational clarity. Real gradient descent typically:
- Uses batches of data (not single examples)
- Runs for many iterations (hundreds or thousands)
- Employs regularization to prevent overfitting
- Uses validation sets to monitor generalization
