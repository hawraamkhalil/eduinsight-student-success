"""Interactive engagement analysis page."""

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
    "Engagement Analysis"
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


result_options = [
    "All",
    *sorted(
        data[
            "final_result"
        ]
        .dropna()
        .unique()
        .tolist()
    ),
]


selected_result = st.selectbox(
    "Filter by final outcome",
    result_options,
)


if selected_result == "All":
    filtered = data
else:
    filtered = data.loc[
        data[
            "final_result"
        ].eq(
            selected_result
        )
    ]


metric_columns = st.columns(3)


metric_columns[0].metric(
    "Students",
    f"{len(filtered):,}",
)

metric_columns[1].metric(
    "Median early clicks",
    f"{filtered['total_clicks_30'].median():,.0f}",
)

metric_columns[2].metric(
    "Median active days",
    f"{filtered['active_days_30'].median():.0f}",
)


st.subheader(
    "Active-day distribution"
)


active_day_counts = (
    filtered[
        "active_days_30"
    ]
    .value_counts()
    .sort_index()
    .rename(
        "Students"
    )
)


st.line_chart(
    active_day_counts
)


st.subheader(
    "Median engagement by outcome"
)


comparison = (
    data
    .groupby(
        "final_result"
    )[
        [
            "total_clicks_30",
            "active_days_30",
            "unique_resources_30",
            "assessments_submitted_30",
        ]
    ]
    .median()
    .round(1)
)


st.dataframe(
    comparison,
    use_container_width=True,
)


st.warning(
    "These are associations in an observational dataset. They do not prove that "
    "a specific activity level causes a particular result."
)