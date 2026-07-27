"""Interactive demonstration for the fitted early-support classifier."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BASE_DIR),
    )


from src.model_training import (
    load_model_artifact,
)


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "early_support_model.joblib"
)


@st.cache_resource
def load_artifact(
    path: Path,
) -> dict:
    return load_model_artifact(
        path
    )


def get_default(
    summary: dict,
    feature: str,
    fallback: float = 0.0,
) -> float:
    return float(
        summary
        .get(
            feature,
            {},
        )
        .get(
            "median",
            fallback,
        )
    )


st.title(
    "Early-Support Demonstration"
)


st.write(
    "Enter a hypothetical student profile using information available during "
    "the first 30 course days."
)


try:
    artifact = load_artifact(
        MODEL_PATH
    )

except (
    FileNotFoundError,
    ValueError,
) as error:
    st.error(str(error))

    st.info(
        "Run notebooks/04_model_development.ipynb "
        "before using this page."
    )

    st.stop()


model = artifact["model"]

feature_columns = artifact[
    "feature_columns"
]

feature_summary = artifact[
    "feature_summary"
]

metadata = artifact[
    "metadata"
]

metrics = artifact[
    "metrics"
]


with st.form(
    "prediction_form"
):
    left, right = st.columns(2)

    with left:
        previous_attempts = (
            st.number_input(
                "Previous attempts",
                min_value=0,
                max_value=20,
                value=int(
                    round(
                        get_default(
                            feature_summary,
                            "num_of_prev_attempts",
                        )
                    )
                ),
            )
        )

        studied_credits = (
            st.number_input(
                "Studied credits",
                min_value=0,
                max_value=400,
                value=int(
                    round(
                        get_default(
                            feature_summary,
                            "studied_credits",
                            60,
                        )
                    )
                ),
            )
        )

        registration_day = (
            st.number_input(
                "Registration day relative to course start",
                min_value=-400,
                max_value=30,
                value=int(
                    round(
                        get_default(
                            feature_summary,
                            "date_registration",
                            -20,
                        )
                    )
                ),
            )
        )

        total_clicks = (
            st.number_input(
                "Total VLE clicks in days 0–30",
                min_value=0,
                value=int(
                    round(
                        get_default(
                            feature_summary,
                            "total_clicks_30",
                            100,
                        )
                    )
                ),
            )
        )

        active_days = (
            st.number_input(
                "Active days in days 0–30",
                min_value=0,
                max_value=31,
                value=min(
                    31,
                    int(
                        round(
                            get_default(
                                feature_summary,
                                "active_days_30",
                                10,
                            )
                        )
                    ),
                ),
            )
        )

        unique_resources = (
            st.number_input(
                "Unique VLE resources accessed",
                min_value=0,
                value=int(
                    round(
                        get_default(
                            feature_summary,
                            "unique_resources_30",
                            10,
                        )
                    )
                ),
            )
        )

    with right:
        submitted_assessments = (
            st.number_input(
                "Assessments submitted by day 30",
                min_value=0,
                max_value=20,
                value=int(
                    round(
                        get_default(
                            feature_summary,
                            "assessments_submitted_30",
                            1,
                        )
                    )
                ),
            )
        )

        no_early_score = (
            st.checkbox(
                "No early assessment score is available",
                value=(
                    submitted_assessments
                    == 0
                ),
            )
        )

        average_score = (
            st.number_input(
                "Average early assessment score",
                min_value=0.0,
                max_value=100.0,
                value=float(
                    get_default(
                        feature_summary,
                        "avg_score_30",
                        60.0,
                    )
                ),
                disabled=(
                    no_early_score
                ),
            )
        )

        late_submissions = (
            st.number_input(
                "Late early submissions",
                min_value=0,
                max_value=20,
                value=0,
                disabled=(
                    submitted_assessments
                    == 0
                ),
            )
        )

    submitted = (
        st.form_submit_button(
            "Generate estimate"
        )
    )


if submitted:
    safe_active_days = int(
        active_days
    )

    average_clicks = (
        total_clicks
        / safe_active_days
        if safe_active_days > 0
        else 0.0
    )

    assessment_count = int(
        submitted_assessments
    )

    early_score = (
        np.nan
        if (
            no_early_score
            or assessment_count == 0
        )
        else float(
            average_score
        )
    )

    late_count = (
        0
        if assessment_count == 0
        else int(
            late_submissions
        )
    )

    input_row = pd.DataFrame(
        [
            {
                "num_of_prev_attempts": previous_attempts,
                "studied_credits": studied_credits,
                "date_registration": registration_day,
                "total_clicks_30": total_clicks,
                "active_days_30": active_days,
                "unique_resources_30": unique_resources,
                "avg_clicks_per_active_day_30": average_clicks,
                "assessments_submitted_30": assessment_count,
                "avg_score_30": early_score,
                "late_submissions_30": late_count,
                "has_early_assessment": int(
                    assessment_count > 0
                ),
            }
        ]
    ).loc[
        :,
        feature_columns,
    ]

    support_probability = float(
        model.predict_proba(
            input_row
        )[0, 1]
    )

    threshold = float(
        metrics.get(
            "threshold",
            0.50,
        )
    )

    if (
        support_probability
        >= threshold
    ):
        st.warning(
            "This profile may benefit from additional academic support."
        )
    else:
        st.success(
            "This profile was not placed in the support flag."
        )

    st.metric(
        "Model support score",
        f"{support_probability:.1%}",
    )

    st.caption(
        f"Model: "
        f"{metadata.get('model_name', 'Not recorded')} · "
        f"Observation cutoff: "
        f"day {metadata.get('cutoff_day', 30)} · "
        f"Decision threshold: "
        f"{threshold:.2f}"
    )


st.info(
    "The score is a portfolio demonstration—not a diagnosis, grade, or automatic "
    "academic decision. Human review and contextual evidence are essential."
)