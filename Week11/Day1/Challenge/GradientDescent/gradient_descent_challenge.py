"""
Predicting Student Exam Scores Using Gradient Descent
======================================================

Educational implementation demonstrating a single step of gradient descent
for a simple neural network that predicts student exam scores based on:
- Study Hours (x1): Number of hours studied for the exam
- Previous Test Score (x2): Score from the last test

This script walks through the complete gradient descent process:
1. Forward pass: Calculate prediction using current weights and bias
2. Loss calculation: Measure error using Mean Squared Error (MSE)
3. Gradient computation: Calculate partial derivatives of loss w.r.t. parameters
4. Parameter update: Apply gradient descent to improve weights and bias

The goal is to show how a neural network learns by iteratively adjusting
its parameters to reduce prediction error, without using ML libraries.

Mathematical formulas:
- Forward pass: ŷ = w1·x1 + w2·x2 + b
- Loss: L = (1/2)(y - ŷ)²
- Gradients: ∂L/∂w1 = -(y - ŷ)·x1, ∂L/∂w2 = -(y - ŷ)·x2, ∂L/∂b = -(y - ŷ)
- Update: θ_new = θ_old - α·∂L/∂θ
"""

from pathlib import Path
from typing import Tuple, Dict, Any

# Constants - Initial neural network parameters
W1_INITIAL = 0.6  # Weight for study hours
W2_INITIAL = 0.3  # Weight for previous test score
B_INITIAL = 10  # Bias term
LEARNING_RATE = 0.01  # Step size for gradient descent (α)

# Sample student data
STUDY_HOURS = 5  # x1: Hours studied for the exam
PREVIOUS_SCORE = 70  # x2: Score from previous test
ACTUAL_SCORE = 85  # y: Actual exam score (target)

# Output configuration
OUTPUT_DIR = Path("outputs")
REPORT_FILENAME = "gradient_descent_report.txt"


def _log(step: str, **details: Any) -> None:
    """
    Emit a structured progress message for the gradient descent process.

    Args:
        step: Description of the current step
        **details: Key-value pairs with additional context
    """
    if details:
        formatted = " | ".join(f"{key}={value}" for key, value in details.items())
        print(f"[FLOW] {step:<35} :: {formatted}")
    else:
        print(f"[FLOW] {step}")


def predict(
    x1: float, x2: float, w1: float, w2: float, b: float
) -> Tuple[float, Dict[str, float]]:
    """
    Perform forward pass to calculate prediction.

    Computes: ŷ = w1·x1 + w2·x2 + b

    Args:
        x1: Study hours input
        x2: Previous test score input
        w1: Weight for x1
        w2: Weight for x2
        b: Bias term

    Returns:
        Tuple containing:
        - prediction: The computed exam score prediction
        - intermediates: Dictionary with intermediate calculation values
    """
    # Calculate weighted sum of inputs
    term1 = w1 * x1  # Contribution from study hours
    term2 = w2 * x2  # Contribution from previous score

    # Add bias and compute final prediction
    prediction = term1 + term2 + b

    intermediates = {
        "w1_x_x1": term1,
        "w2_x_x2": term2,
        "sum_before_bias": term1 + term2,
        "prediction": prediction,
    }

    return prediction, intermediates


def calculate_loss(actual: float, predicted: float) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Mean Squared Error (MSE) loss.

    Computes: L = (1/2)(y - ŷ)²

    The factor of 1/2 simplifies the derivative calculation.

    Args:
        actual: True exam score
        predicted: Predicted exam score

    Returns:
        Tuple containing:
        - loss: The computed MSE loss value
        - intermediates: Dictionary with error and squared error
    """
    error = actual - predicted  # Prediction error (y - ŷ)
    squared_error = error**2  # Square the error
    loss = 0.5 * squared_error  # Apply 1/2 factor for derivative convenience

    intermediates = {"error": error, "squared_error": squared_error, "loss": loss}

    return loss, intermediates


def compute_gradients(
    x1: float, x2: float, actual: float, predicted: float
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Calculate gradients (partial derivatives) of loss w.r.t. parameters.

    Using chain rule:
    - ∂L/∂w1 = ∂L/∂ŷ · ∂ŷ/∂w1 = -(y - ŷ) · x1
    - ∂L/∂w2 = ∂L/∂ŷ · ∂ŷ/∂w2 = -(y - ŷ) · x2
    - ∂L/∂b = ∂L/∂ŷ · ∂ŷ/∂b = -(y - ŷ) · 1 = -(y - ŷ)

    Gradients indicate direction and magnitude of steepest increase in loss.
    We'll move in the opposite direction (negative gradient) to minimize loss.

    Args:
        x1: Study hours input
        x2: Previous test score input
        actual: True exam score
        predicted: Predicted exam score

    Returns:
        Tuple containing:
        - grad_w1: Gradient w.r.t. w1
        - grad_w2: Gradient w.r.t. w2
        - grad_b: Gradient w.r.t. bias
        - intermediates: Dictionary with calculation details
    """
    error = actual - predicted  # y - ŷ
    neg_error = -error  # -(y - ŷ), common term in all gradients

    # Calculate partial derivatives
    grad_w1 = neg_error * x1  # ∂L/∂w1
    grad_w2 = neg_error * x2  # ∂L/∂w2
    grad_b = neg_error  # ∂L/∂b

    intermediates = {
        "error": error,
        "neg_error": neg_error,
        "grad_w1": grad_w1,
        "grad_w2": grad_w2,
        "grad_b": grad_b,
    }

    return grad_w1, grad_w2, grad_b, intermediates


def update_parameters(
    w1: float,
    w2: float,
    b: float,
    grad_w1: float,
    grad_w2: float,
    grad_b: float,
    learning_rate: float,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Update parameters using gradient descent rule.

    Applies: θ_new = θ_old - α · ∂L/∂θ

    Where:
    - θ represents each parameter (w1, w2, b)
    - α is the learning rate (step size)
    - ∂L/∂θ is the gradient (direction of steepest increase)

    By subtracting α·gradient, we move parameters in direction that reduces loss.

    Args:
        w1: Current weight for x1
        w2: Current weight for x2
        b: Current bias
        grad_w1: Gradient w.r.t. w1
        grad_w2: Gradient w.r.t. w2
        grad_b: Gradient w.r.t. bias
        learning_rate: Step size (α)

    Returns:
        Tuple containing:
        - w1_new: Updated weight for x1
        - w2_new: Updated weight for x2
        - b_new: Updated bias
        - intermediates: Dictionary with update calculations
    """
    # Calculate parameter updates (changes to be applied)
    delta_w1 = learning_rate * grad_w1
    delta_w2 = learning_rate * grad_w2
    delta_b = learning_rate * grad_b

    # Apply updates to get new parameters
    w1_new = w1 - delta_w1
    w2_new = w2 - delta_w2
    b_new = b - delta_b

    intermediates = {
        "delta_w1": delta_w1,
        "delta_w2": delta_w2,
        "delta_b": delta_b,
        "w1_new": w1_new,
        "w2_new": w2_new,
        "b_new": b_new,
    }

    return w1_new, w2_new, b_new, intermediates


def generate_report(
    x1: float,
    x2: float,
    y: float,
    w1_old: float,
    w2_old: float,
    b_old: float,
    pred_old: float,
    pred_intermediates: Dict[str, float],
    loss: float,
    loss_intermediates: Dict[str, float],
    grads: Tuple[float, float, float],
    grad_intermediates: Dict[str, float],
    w1_new: float,
    w2_new: float,
    b_new: float,
    update_intermediates: Dict[str, float],
    pred_new: float,
    loss_new: float,
    learning_rate: float,
    output_path: Path,
) -> None:
    """
    Generate detailed text report showing all gradient descent calculations.

    Creates a comprehensive report with:
    - Initial parameters and input data
    - Forward pass calculations
    - Loss computation
    - Gradient calculations with formulas
    - Parameter updates
    - Before/after comparison

    Args:
        x1: Study hours input
        x2: Previous test score input
        y: Actual exam score
        w1_old: Initial weight for x1
        w2_old: Initial weight for x2
        b_old: Initial bias
        pred_old: Initial prediction
        pred_intermediates: Forward pass calculation details
        loss: Initial loss value
        loss_intermediates: Loss calculation details
        grads: Tuple of (grad_w1, grad_w2, grad_b)
        grad_intermediates: Gradient calculation details
        w1_new: Updated weight for x1
        w2_new: Updated weight for x2
        b_new: Updated bias
        update_intermediates: Parameter update details
        pred_new: Prediction after update
        loss_new: Loss after update
        learning_rate: Learning rate used
        output_path: Path to save the report
    """
    grad_w1, grad_w2, grad_b = grads

    lines = [
        "=" * 80,
        "GRADIENT DESCENT FOR STUDENT EXAM SCORE PREDICTION",
        "=" * 80,
        "",
        "PROBLEM STATEMENT",
        "-" * 80,
        "A school wants to predict students' final exam scores based on:",
        "  • Study Hours (x1): Number of hours studied",
        "  • Previous Test Score (x2): Score from last test",
        "",
        "We will perform ONE step of gradient descent to improve our neural network.",
        "",
        "",
        "=" * 80,
        "STEP 1: INITIAL SETUP",
        "=" * 80,
        "",
        "Input Data (Sample Student):",
        f"  x1 (Study Hours)      = {x1}",
        f"  x2 (Previous Score)   = {x2}",
        f"  y (Actual Exam Score) = {y}",
        "",
        "Initial Neural Network Parameters:",
        f"  w1 (Weight for x1) = {w1_old}",
        f"  w2 (Weight for x2) = {w2_old}",
        f"  b (Bias)           = {b_old}",
        "",
        "Learning Rate:",
        f"  α (Alpha) = {learning_rate}",
        "",
        "",
        "=" * 80,
        "STEP 2: FORWARD PASS (Make Prediction)",
        "=" * 80,
        "",
        "Formula: ŷ = w1·x1 + w2·x2 + b",
        "",
        "Calculation:",
        f"  w1 · x1 = {w1_old} × {x1} = {pred_intermediates['w1_x_x1']:.2f}",
        f"  w2 · x2 = {w2_old} × {x2} = {pred_intermediates['w2_x_x2']:.2f}",
        f"  Sum     = {pred_intermediates['w1_x_x1']:.2f} + {pred_intermediates['w2_x_x2']:.2f} = {pred_intermediates['sum_before_bias']:.2f}",
        f"  Add bias: {pred_intermediates['sum_before_bias']:.2f} + {b_old} = {pred_old:.2f}",
        "",
        f"Prediction: ŷ = {pred_old:.2f}",
        f"Actual:     y = {y}",
        f"Error:      y - ŷ = {y} - {pred_old:.2f} = {loss_intermediates['error']:.2f}",
        "",
        "",
        "=" * 80,
        "STEP 3: CALCULATE LOSS (Measure Error)",
        "=" * 80,
        "",
        "Formula: L = (1/2)(y - ŷ)²",
        "",
        "The factor of 1/2 simplifies gradient calculation.",
        "",
        "Calculation:",
        f"  Error         = y - ŷ = {y} - {pred_old:.2f} = {loss_intermediates['error']:.2f}",
        f"  Squared Error = ({loss_intermediates['error']:.2f})² = {loss_intermediates['squared_error']:.2f}",
        f"  Loss          = (1/2) × {loss_intermediates['squared_error']:.2f} = {loss:.4f}",
        "",
        f"Loss: L = {loss:.4f}",
        "",
        "",
        "=" * 80,
        "STEP 4: COMPUTE GRADIENTS (Direction to Improve)",
        "=" * 80,
        "",
        "Gradients tell us how to change each parameter to reduce loss.",
        "Using chain rule from calculus:",
        "",
        "Formulas:",
        "  ∂L/∂w1 = -(y - ŷ) · x1",
        "  ∂L/∂w2 = -(y - ŷ) · x2",
        "  ∂L/∂b  = -(y - ŷ)",
        "",
        "Calculation:",
        f"  Error         = y - ŷ = {grad_intermediates['error']:.2f}",
        f"  Negative error = -(y - ŷ) = {grad_intermediates['neg_error']:.2f}",
        "",
        f"  ∂L/∂w1 = {grad_intermediates['neg_error']:.2f} × {x1} = {grad_w1:.2f}",
        f"  ∂L/∂w2 = {grad_intermediates['neg_error']:.2f} × {x2} = {grad_w2:.2f}",
        f"  ∂L/∂b  = {grad_intermediates['neg_error']:.2f}",
        "",
        "Gradients:",
        f"  ∂L/∂w1 = {grad_w1:.2f}",
        f"  ∂L/∂w2 = {grad_w2:.2f}",
        f"  ∂L/∂b  = {grad_b:.2f}",
        "",
        "",
        "=" * 80,
        "STEP 5: UPDATE PARAMETERS (Learn from Error)",
        "=" * 80,
        "",
        "Formula: θ_new = θ_old - α · ∂L/∂θ",
        "",
        "We move parameters in opposite direction of gradient (negative gradient)",
        "to reduce loss. Learning rate α controls step size.",
        "",
        "Calculation:",
        f"  Δw1 = α × ∂L/∂w1 = {learning_rate} × {grad_w1:.2f} = {update_intermediates['delta_w1']:.4f}",
        f"  Δw2 = α × ∂L/∂w2 = {learning_rate} × {grad_w2:.2f} = {update_intermediates['delta_w2']:.4f}",
        f"  Δb  = α × ∂L/∂b  = {learning_rate} × {grad_b:.2f} = {update_intermediates['delta_b']:.4f}",
        "",
        f"  w1_new = {w1_old} - {update_intermediates['delta_w1']:.4f} = {w1_new:.4f}",
        f"  w2_new = {w2_old} - {update_intermediates['delta_w2']:.4f} = {w2_new:.4f}",
        f"  b_new  = {b_old} - {update_intermediates['delta_b']:.4f} = {b_new:.4f}",
        "",
        "Updated Parameters:",
        f"  w1 = {w1_new:.4f}  (changed by {w1_new - w1_old:+.4f})",
        f"  w2 = {w2_new:.4f}  (changed by {w2_new - w2_old:+.4f})",
        f"  b  = {b_new:.4f}  (changed by {b_new - b_old:+.4f})",
        "",
        "",
        "=" * 80,
        "STEP 6: VERIFY IMPROVEMENT",
        "=" * 80,
        "",
        "Let's check if the updated parameters give a better prediction:",
        "",
        "BEFORE UPDATE:",
        f"  Parameters: w1={w1_old}, w2={w2_old}, b={b_old}",
        f"  Prediction: ŷ = {pred_old:.2f}",
        f"  Loss:       L = {loss:.4f}",
        f"  Error:      |y - ŷ| = |{y} - {pred_old:.2f}| = {abs(y - pred_old):.2f}",
        "",
        "AFTER UPDATE:",
        f"  Parameters: w1={w1_new:.4f}, w2={w2_new:.4f}, b={b_new:.4f}",
        f"  Prediction: ŷ = {pred_new:.2f}",
        f"  Loss:       L = {loss_new:.4f}",
        f"  Error:      |y - ŷ| = |{y} - {pred_new:.2f}| = {abs(y - pred_new):.2f}",
        "",
        "IMPROVEMENT:",
        f"  Loss change: {loss - loss_new:+.4f} ({'decreased' if loss_new < loss else 'increased'})",
        f"  Prediction error change: {abs(y - pred_old) - abs(y - pred_new):+.2f}",
        f"  Prediction moved {'closer to' if abs(y - pred_new) < abs(y - pred_old) else 'further from'} actual by: {abs(abs(y - pred_new) - abs(y - pred_old)):.2f} points",
        "",
        "",
        "=" * 80,
        "SUMMARY & OBSERVATIONS",
        "=" * 80,
        "",
        "In this single gradient descent step, we:",
        "  1. Made a prediction using initial weights and bias",
        "  2. Calculated how wrong our prediction was (loss)",
        "  3. Computed gradients showing how to adjust parameters",
        "  4. Updated parameters by moving opposite to gradient direction",
        "  5. Verified the result of our parameter update",
        "",
        "IMPORTANT OBSERVATIONS:",
        "",
        "The gradients were all negative, which means:",
        "  • Our prediction (34) was too LOW compared to actual (85)",
        "  • We need to INCREASE the weights and bias",
        "  • Subtracting negative gradients = adding positive values ✓",
        "",
        f"After the update, prediction changed from {pred_old:.2f} to {pred_new:.2f}.",
        "",
        "NOTE: In this example, the prediction actually got worse! This happens because:",
        "  • The learning rate (0.01) combined with large gradients caused overshooting",
        "  • Real neural networks normalize/scale input features to prevent this",
        "  • Multiple small steps work better than one large jump",
        "  • Feature scaling (e.g., x2/100) would improve convergence",
        "",
        "This demonstrates why hyperparameter tuning (learning rate, feature scaling)",
        "and proper data preprocessing are crucial in machine learning!",
        "",
        "In practice, gradient descent would:",
        "  • Use normalized features (0-1 range or standardized)",
        "  • Process many examples (batch/mini-batch gradient descent)",
        "  • Run for many iterations until loss converges",
        "  • Use techniques like momentum or adaptive learning rates",
        "",
        "=" * 80,
    ]

    report_text = "\n".join(lines)

    # Save to file
    output_path.write_text(report_text, encoding="utf-8")

    # Also print to console for immediate feedback
    print("\n" + report_text)


def main() -> None:
    """
    Main function to orchestrate the gradient descent demonstration.

    Performs a complete gradient descent step and generates detailed report
    showing all calculations and improvements.
    """
    _log("Gradient Descent Challenge Started")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log("Output directory ready", path=str(OUTPUT_DIR))

    # Step 1: Initial setup
    _log("Initial setup", x1=STUDY_HOURS, x2=PREVIOUS_SCORE, y=ACTUAL_SCORE)
    _log(
        "Initial parameters",
        w1=W1_INITIAL,
        w2=W2_INITIAL,
        b=B_INITIAL,
        alpha=LEARNING_RATE,
    )

    # Step 2: Forward pass with initial parameters
    _log("Performing forward pass")
    pred_old, pred_intermediates = predict(
        STUDY_HOURS, PREVIOUS_SCORE, W1_INITIAL, W2_INITIAL, B_INITIAL
    )
    _log("Prediction computed", prediction=f"{pred_old:.2f}", actual=ACTUAL_SCORE)

    # Step 3: Calculate loss
    _log("Calculating loss")
    loss, loss_intermediates = calculate_loss(ACTUAL_SCORE, pred_old)
    _log(
        "Loss computed", loss=f"{loss:.4f}", error=f"{loss_intermediates['error']:.2f}"
    )

    # Step 4: Compute gradients
    _log("Computing gradients")
    grad_w1, grad_w2, grad_b, grad_intermediates = compute_gradients(
        STUDY_HOURS, PREVIOUS_SCORE, ACTUAL_SCORE, pred_old
    )
    _log(
        "Gradients computed",
        grad_w1=f"{grad_w1:.2f}",
        grad_w2=f"{grad_w2:.2f}",
        grad_b=f"{grad_b:.2f}",
    )

    # Step 5: Update parameters
    _log("Updating parameters")
    w1_new, w2_new, b_new, update_intermediates = update_parameters(
        W1_INITIAL, W2_INITIAL, B_INITIAL, grad_w1, grad_w2, grad_b, LEARNING_RATE
    )
    _log("Parameters updated", w1=f"{w1_new:.4f}", w2=f"{w2_new:.4f}", b=f"{b_new:.4f}")

    # Step 6: Verify improvement with new parameters
    _log("Verifying improvement")
    pred_new, _ = predict(STUDY_HOURS, PREVIOUS_SCORE, w1_new, w2_new, b_new)
    loss_new, _ = calculate_loss(ACTUAL_SCORE, pred_new)
    _log(
        "New prediction computed", prediction=f"{pred_new:.2f}", loss=f"{loss_new:.4f}"
    )

    improvement = loss - loss_new
    _log("Improvement verified", loss_reduction=f"{improvement:.4f}")

    # Generate comprehensive report
    _log("Generating detailed report")
    report_path = OUTPUT_DIR / REPORT_FILENAME
    generate_report(
        STUDY_HOURS,
        PREVIOUS_SCORE,
        ACTUAL_SCORE,
        W1_INITIAL,
        W2_INITIAL,
        B_INITIAL,
        pred_old,
        pred_intermediates,
        loss,
        loss_intermediates,
        (grad_w1, grad_w2, grad_b),
        grad_intermediates,
        w1_new,
        w2_new,
        b_new,
        update_intermediates,
        pred_new,
        loss_new,
        LEARNING_RATE,
        report_path,
    )

    _log("Report saved", path=str(report_path))
    _log("Gradient Descent Challenge Complete")


if __name__ == "__main__":
    main()
