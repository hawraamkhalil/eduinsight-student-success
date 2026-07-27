"""EduInsight Streamlit application home page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "student_features.csv"
)


st.set_page_config(
    page_title="EduInsight",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_processed_data(
    path: Path,
) -> pd.DataFrame:
    """Load the prepared feature table once per application session."""

    if not path.is_file():
        raise FileNotFoundError(
            "Processed data is missing. Run "
            "notebooks/02_data_preparation.ipynb "
            "before launching the dashboard."
        )

    return pd.read_csv(path)


st.title("EduInsight")

st.subheader(
    "Student Engagement and "
    "Early-Support Analytics"
)

st.write(
    "EduInsight explores whether activity observed during the first 30 days of "
    "a course is associated with later student outcomes."
)


try:
    data = load_processed_data(
        DATA_PATH
    )

except (
    FileNotFoundError,
    pd.errors.ParserError,
) as error:
    st.error(str(error))

    st.code(
        "jupyter lab\n"
        "# Run notebooks 01 and 02, then:\n"
        "streamlit run app.py"
    )

    st.stop()


required_columns = {
    "id_student",
    "active_days_30",
    "total_clicks_30",
    "support_needed",
    "final_result",
}


missing_columns = (
    required_columns
    - set(data.columns)
)


if missing_columns:
    st.error(
        "The processed file does not match the expected schema. Missing: "
        + ", ".join(
            sorted(
                missing_columns
            )
        )
    )

    st.stop()


student_count = int(
    data[
        "id_student"
    ].nunique()
)

support_rate = float(
    data[
        "support_needed"
    ].mean() * 100
)

average_active_days = float(
    data[
        "active_days_30"
    ].mean()
)

average_clicks = float(
    data[
        "total_clicks_30"
    ].mean()
)


metric_columns = st.columns(4)


metric_columns[0].metric(
    "Students analysed",
    f"{student_count:,}",
)

metric_columns[1].metric(
    "Average active days",
    f"{average_active_days:.1f}",
)

metric_columns[2].metric(
    "Average early clicks",
    f"{average_clicks:,.0f}",
)

metric_columns[3].metric(
    "Support-flag category",
    f"{support_rate:.1f}%",
)


st.divider()


left_column, right_column = (
    st.columns(
        [1.35, 1]
    )
)


with left_column:
    st.markdown(
        "### What the project demonstrates"
    )

    st.markdown(
        """
        - Multi-table data preparation with explicit join validation
        - Memory-conscious processing of the large VLE activity file
        - Leakage-aware early feature engineering
        - Cross-validated model selection and holdout evaluation
        - A deployable Streamlit interface
        """
    )


with right_column:
    st.markdown(
        "### Outcome distribution"
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

    st.bar_chart(
        outcome_counts
    )


st.info(
    "This is a portfolio and learning-analytics demonstration. It must not be "
    "used as the sole basis for real academic decisions."
)