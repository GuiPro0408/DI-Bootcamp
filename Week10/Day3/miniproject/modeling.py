"""Modeling and evaluation utilities."""
from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from config import RANDOM_STATE


def _log(step: str, **details: object) -> None:
    """Emit structured modeling diagnostics."""
    if details:
        formatted = " | ".join(f"{k}={v}" for k, v in details.items())
        print(f"[MODEL] {step:<26} :: {formatted}")
    else:
        print(f"[MODEL] {step}")


def build_preprocessor(
        numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric scaling and categorical encoding.

    Args:
        numeric_cols: Numeric feature names to scale with MinMaxScaler.
        categorical_cols: Categorical feature names to encode with OneHotEncoder.

    Returns:
        A ColumnTransformer suitable for fitting within scikit-learn pipelines.
    """
    _log("Building preprocessor", numeric=len(numeric_cols), categorical=len(categorical_cols))
    transformer = ColumnTransformer([
        ("num", MinMaxScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_cols),
    ])
    return transformer


def train_logreg_with_grid(
        x_train, y_train, preprocessor
) -> GridSearchCV:
    """Train a logistic regression model with hyper-parameter tuning.

    Args:
        x_train: Training feature dataframe or array.
        y_train: Training labels.
        preprocessor: Pre-fitted preprocessing transformer to include in the pipeline.

    Returns:
        A fitted GridSearchCV instance configured to optimise F1 score.
    """
    pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
    param_grid = {
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l2"],
        "clf__C": [0.1, 0.5, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=1, refit=True, verbose=0)
    _log("Starting GridSearchCV", samples=len(y_train))
    grid.fit(x_train, y_train)
    _log("Grid search finished", best_score=round(grid.best_score_, 4))
    return grid


def evaluate(
        model, x_test, y_test
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute evaluation metrics and confusion matrix for a fitted classifier.

    Args:
        model: Trained estimator exposing a predict method.
        x_test: Test features.
        y_test: Ground-truth labels for the test set.

    Returns:
        A tuple containing a metrics dictionary and the confusion matrix array.
    """
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "class0": {
            "precision": float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        },
        "class1": {
            "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)
    _log(
        "Evaluation metrics computed",
        accuracy=round(metrics["accuracy"], 4),
        f1_pos=round(metrics["class1"]["f1"], 4),
        samples=len(y_test),
    )
    return metrics, cm


def _import_bonus():
    """Import optional classifiers lazily to avoid heavy imports when unused."""
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    return SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier


def run_bonus_suite(
        x_train, x_test, y_train, y_test, preprocessor
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a suite of additional classifiers with basic hyper-parameter grids.

    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        preprocessor: Shared preprocessing transformer to reuse across models.

    Returns:
        Mapping from model name to a summary of best parameters and test metrics.
    """
    svc, k_neighbors_classifier, decision_tree_classifier, random_forest_classifier, gradient_boosting_classifier = _import_bonus()

    _log("Running bonus suite", train=len(y_train), test=len(y_test))
    configs = {
        "SVM": (Pipeline([("pre", preprocessor), ("clf", svc())]), {"clf__C": [0.5, 1.0], "clf__kernel": ["rbf", "linear"]}),
        "KNN": (Pipeline([("pre", preprocessor), ("clf", k_neighbors_classifier())]), {"clf__n_neighbors": [3, 5]}),
        "DecisionTree": (Pipeline([("pre", preprocessor), ("clf", decision_tree_classifier(random_state=RANDOM_STATE))]), {"clf__max_depth": [None, 5, 10]}),
        "RandomForest": (Pipeline([("pre", preprocessor), ("clf", random_forest_classifier(random_state=RANDOM_STATE))]), {"clf__n_estimators": [100, 300]}),
        "GradientBoosting": (Pipeline([("pre", preprocessor), ("clf", gradient_boosting_classifier(random_state=RANDOM_STATE))]), {"clf__n_estimators": [100], "clf__learning_rate": [0.1]}),
    }
    out: Dict[str, Dict[str, Any]] = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name, (pipe, grid_params) in configs.items():
        _log("Tuning bonus model", model=name)
        grid = GridSearchCV(pipe, grid_params, scoring="f1", cv=cv, n_jobs=1, refit=True, verbose=0)
        grid.fit(x_train, y_train)
        y_pred = grid.predict(x_test)
        out[name] = {
            "best_params": grid.best_params_,
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_pos": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        }
        _log(
            "Bonus model scored",
            model=name,
            accuracy=round(out[name]["test_accuracy"], 4),
            f1_pos=round(out[name]["test_f1_pos"], 4),
        )
    return out
