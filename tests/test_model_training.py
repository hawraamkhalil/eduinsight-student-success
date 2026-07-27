from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_training import (
    MODEL_FEATURES,
    build_candidate_models,
    create_holdout_split,
    cross_validate_candidates,
    evaluate_classifier,
)


def make_modelling_data(
    row_count: int = 120,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        42
    )

    support_needed = np.array(
        [0, 1]
        * (
            row_count // 2
        ),
        dtype=int,
    )

    rng.shuffle(
        support_needed
    )

    data = pd.DataFrame(
        {
            "num_of_prev_attempts": (
                rng.integers(
                    0,
                    4,
                    row_count,
                )
            ),
            "studied_credits": (
                rng.integers(
                    30,
                    121,
                    row_count,
                )
            ),
            "date_registration": (
                rng.integers(
                    -80,
                    10,
                    row_count,
                )
            ),
            "total_clicks_30": (
                rng.integers(
                    0,
                    400,
                    row_count,
                )
            ),
            "active_days_30": (
                rng.integers(
                    0,
                    31,
                    row_count,
                )
            ),
            "unique_resources_30": (
                rng.integers(
                    0,
                    50,
                    row_count,
                )
            ),
            "avg_clicks_per_active_day_30": (
                rng.uniform(
                    0,
                    30,
                    row_count,
                )
            ),
            "assessments_submitted_30": (
                rng.integers(
                    0,
                    4,
                    row_count,
                )
            ),
            "avg_score_30": (
                rng.uniform(
                    20,
                    100,
                    row_count,
                )
            ),
            "late_submissions_30": (
                rng.integers(
                    0,
                    3,
                    row_count,
                )
            ),
            "has_early_assessment": (
                rng.integers(
                    0,
                    2,
                    row_count,
                )
            ),
            "support_needed": (
                support_needed
            ),
        }
    )

    data.loc[
        data[
            "has_early_assessment"
        ].eq(0),
        "avg_score_30",
    ] = np.nan

    return data


def test_model_selection_and_evaluation_workflow() -> None:
    data = make_modelling_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_holdout_split(
        data
    )

    candidates = (
        build_candidate_models(
            random_state=42
        )
    )

    comparison = (
        cross_validate_candidates(
            candidates,
            X_train,
            y_train,
            n_splits=3,
        )
    )

    assert set(
        comparison[
            "model"
        ]
    ) == {
        "Logistic Regression",
        "Random Forest",
    }

    selected_name = (
        comparison.iloc[0][
            "model"
        ]
    )

    selected_model = (
        candidates[
            selected_name
        ]
    )

    selected_model.fit(
        X_train.loc[
            :,
            list(MODEL_FEATURES),
        ],
        y_train,
    )

    metrics = (
        evaluate_classifier(
            selected_model,
            X_test,
            y_test,
        )
    )

    assert (
        0.0
        <= metrics[
            "f1_support"
        ]
        <= 1.0
    )

    assert len(
        metrics[
            "confusion_matrix"
        ]
    ) == 2