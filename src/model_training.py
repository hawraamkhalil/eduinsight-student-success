"""Model construction, evaluation, selection, and persistence utilities."""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.base import (
    ClassifierMixin,
)
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.impute import (
    SimpleImputer,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)


MODEL_FEATURES: tuple[str, ...] = (
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "total_clicks_30",
    "active_days_30",
    "unique_resources_30",
    "avg_clicks_per_active_day_30",
    "assessments_submitted_30",
    "avg_score_30",
    "late_submissions_30",
    "has_early_assessment",
)


TARGET_COLUMN = "support_needed"


def validate_modelling_data(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[
        str
    ] = MODEL_FEATURES,
    target_column: str = (
        TARGET_COLUMN
    ),
) -> None:
    """Validate required columns and a usable binary target."""

    required = (
        set(feature_columns)
        | {target_column}
    )

    missing = sorted(
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Modelling data is missing columns: "
            f"{', '.join(missing)}"
        )

    target_values = set(
        dataframe[
            target_column
        ]
        .dropna()
        .unique()
    )

    if (
        not target_values.issubset(
            {0, 1}
        )
        or len(target_values) != 2
    ):
        raise ValueError(
            f"{target_column} must be a "
            "non-empty binary target "
            "containing 0 and 1."
        )


def build_candidate_models(
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Create one interpretable baseline and one nonlinear comparison model."""

    logistic_regression = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight=(
                        "balanced"
                    ),
                    random_state=(
                        random_state
                    ),
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight=(
                        "balanced"
                    ),
                    random_state=(
                        random_state
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": (
            logistic_regression
        ),
        "Random Forest": (
            random_forest
        ),
    }


def create_holdout_split(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[
        str
    ] = MODEL_FEATURES,
    target_column: str = (
        TARGET_COLUMN
    ),
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create a stratified holdout split that is untouched during model selection."""

    validate_modelling_data(
        dataframe,
        feature_columns,
        target_column,
    )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    X = dataframe.loc[
        :,
        list(feature_columns),
    ].copy()

    y = (
        dataframe[
            target_column
        ]
        .astype("int8")
        .copy()
    )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def cross_validate_candidates(
    models: Mapping[
        str,
        ClassifierMixin,
    ],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare candidate models using only the training partition."""

    minimum_class_size = int(
        y_train.value_counts().min()
    )

    effective_splits = min(
        n_splits,
        minimum_class_size,
    )

    if effective_splits < 2:
        raise ValueError(
            "At least two examples from each target class are required for "
            "stratified cross-validation."
        )

    cross_validator = (
        StratifiedKFold(
            n_splits=(
                effective_splits
            ),
            shuffle=True,
            random_state=(
                random_state
            ),
        )
    )

    scoring = {
        "f1_support": "f1",
        "roc_auc": "roc_auc",
        "average_precision": (
            "average_precision"
        ),
        "balanced_accuracy": (
            "balanced_accuracy"
        ),
    }

    rows: list[
        dict[str, float | str]
    ] = []

    for model_name, model in (
        models.items()
    ):
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=None,
            error_score="raise",
        )

        rows.append(
            {
                "model": model_name,
                "mean_cv_f1_support": float(
                    scores[
                        "test_f1_support"
                    ].mean()
                ),
                "mean_cv_roc_auc": float(
                    scores[
                        "test_roc_auc"
                    ].mean()
                ),
                "mean_cv_average_precision": float(
                    scores[
                        "test_average_precision"
                    ].mean()
                ),
                "mean_cv_balanced_accuracy": float(
                    scores[
                        "test_balanced_accuracy"
                    ].mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "mean_cv_f1_support",
                "mean_cv_average_precision",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def evaluate_classifier(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate a fitted binary classifier on the untouched holdout set."""

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "The fitted model must implement predict_proba()."
        )

    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prediction = (
        probability >= threshold
    ).astype("int8")

    return {
        "threshold": threshold,
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                prediction,
            )
        ),
        "precision_support": float(
            precision_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "recall_support": float(
            recall_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "f1_support": float(
            f1_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_test,
                probability,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_test,
                prediction,
            ).tolist()
        ),
        "classification_report": (
            classification_report(
                y_test,
                prediction,
                target_names=[
                    "No support flag",
                    "Support flag",
                ],
                output_dict=True,
                zero_division=0,
            )
        ),
    }


def build_feature_summary(
    X_train: pd.DataFrame,
    feature_columns: Sequence[
        str
    ] = MODEL_FEATURES,
) -> dict[
    str,
    dict[str, float],
]:
    """Create safe numeric defaults and ranges for the Streamlit demonstration."""

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for column in feature_columns:
        values = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )

        non_missing = (
            values.dropna()
        )

        if non_missing.empty:
            summary[column] = {
                "min": 0.0,
                "median": 0.0,
                "max": 0.0,
            }

            continue

        summary[column] = {
            "min": float(
                non_missing.min()
            ),
            "median": float(
                non_missing.median()
            ),
            "max": float(
                non_missing.max()
            ),
        }

    return summary


def save_model_artifact(
    model: ClassifierMixin,
    output_path: Path,
    *,
    feature_columns: Sequence[
        str
    ],
    metadata: Mapping[
        str,
        Any,
    ],
    metrics: Mapping[
        str,
        Any,
    ],
    feature_summary: Mapping[
        str,
        Mapping[str, float],
    ],
) -> None:
    """Persist the fitted pipeline together with reproducibility metadata."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "feature_columns": list(
            feature_columns
        ),
        "metadata": dict(
            metadata
        ),
        "metrics": dict(
            metrics
        ),
        "feature_summary": {
            key: dict(value)
            for key, value
            in feature_summary.items()
        },
    }

    joblib.dump(
        artifact,
        output_path,
    )


def load_model_artifact(
    model_path: Path,
) -> dict[str, Any]:
    """Load and validate a saved model artifact."""

    model_path = Path(
        model_path
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: "
            f"{model_path}"
        )

    artifact = joblib.load(
        model_path
    )

    required_keys = {
        "model",
        "feature_columns",
        "metadata",
        "metrics",
        "feature_summary",
    }

    missing = (
        required_keys
        - set(artifact)
    )

    if missing:
        raise ValueError(
            "Model artifact is missing keys: "
            f"{', '.join(sorted(missing))}"
        )

    return artifact