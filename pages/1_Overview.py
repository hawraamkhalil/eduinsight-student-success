"""Dataset overview page for the EduInsight Streamlit application."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "student_features.csv"
)


@st.cache_data
def load_data(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            "Run notebooks/02_data_preparation.ipynb "
            "to create the processed data."
        )

    return pd.read_csv(path)


st.title(
    "Dataset Overview"
)


try:
    data = load_data(
        DATA_PATH
    )

except (
    FileNotFoundError,
    pd.errors.ParserError,
) as error:
    st.error(str(error))
    st.stop()


module = (
    data[
        "code_module"
    ].iloc[0]
)

presentation = (
    data[
        "code_presentation"
    ].iloc[0]
)


st.write(
    f"This page summarizes module **{module}** "
    f"in presentation **{presentation}** "
    "after applying the day-30 eligibility rules."
)


outcome_counts = (
    data[
        "final_result"
    ]
    .value_counts()
    .rename_axis(
        "Final result"
    )
    .to_frame(
        "Students"
    )
)


st.subheader(
    "Final outcomes"
)

st.bar_chart(
    outcome_counts
)


summary_columns = [
    "total_clicks_30",
    "active_days_30",
    "unique_resources_30",
    "assessments_submitted_30",
    "avg_score_30",
    "late_submissions_30",
]


available_summary_columns = [
    column
    for column in summary_columns
    if column in data.columns
]


st.subheader(
    "Early-window summary statistics"
)


st.dataframe(
    data[
        available_summary_columns
    ]
    .describe()
    .round(2),
    use_container_width=True,
)


st.caption(
    "The public dashboard intentionally does not display individual student IDs."
)