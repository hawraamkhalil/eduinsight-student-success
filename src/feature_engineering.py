"""Feature engineering for a day-N student early-support experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preparation import (
    STUDENT_KEY,
    require_columns,
    select_presentation,
    validate_unique_key,
)


VLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "total_clicks_30",
    "active_days_30",
    "unique_resources_30",
    "avg_clicks_per_active_day_30",
)


ASSESSMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "assessments_submitted_30",
    "avg_score_30",
    "late_submissions_30",
    "has_early_assessment",
)


def _validate_cutoff_day(
    cutoff_day: int,
) -> None:
    if cutoff_day < 0:
        raise ValueError(
            "cutoff_day must be zero or a positive integer."
        )


def _empty_vle_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *STUDENT_KEY,
            *VLE_FEATURE_COLUMNS,
        ]
    )


def _empty_assessment_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *STUDENT_KEY,
            *ASSESSMENT_FEATURE_COLUMNS,
        ]
    )


def build_early_vle_features(
    student_vle_path: Path,
    module: str,
    presentation: str,
    cutoff_day: int,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Aggregate VLE activity observed from course day 0 through ``cutoff_day``.

    The large `studentVle.csv` file is read in chunks. Only rows for the selected
    presentation and early observation window are retained before aggregation.
    """

    _validate_cutoff_day(cutoff_day)

    student_vle_path = Path(
        student_vle_path
    )

    if not student_vle_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: "
            f"{student_vle_path}"
        )

    if chunksize <= 0:
        raise ValueError(
            "chunksize must be a positive integer."
        )

    usecols = [
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ]

    filtered_chunks: list[
        pd.DataFrame
    ] = []

    for chunk in pd.read_csv(
        student_vle_path,
        usecols=usecols,
        chunksize=chunksize,
    ):
        filtered = chunk.loc[
            chunk["code_module"].eq(module)
            & chunk[
                "code_presentation"
            ].eq(presentation)
            & chunk["date"].between(
                0,
                cutoff_day,
                inclusive="both",
            )
        ].copy()

        if not filtered.empty:
            filtered_chunks.append(
                filtered
            )

    if not filtered_chunks:
        return _empty_vle_features()

    early_vle = pd.concat(
        filtered_chunks,
        ignore_index=True,
    )

    features = (
        early_vle
        .groupby(
            list(STUDENT_KEY),
            as_index=False,
        )
        .agg(
            total_clicks_30=(
                "sum_click",
                "sum",
            ),
            active_days_30=(
                "date",
                "nunique",
            ),
            unique_resources_30=(
                "id_site",
                "nunique",
            ),
        )
        .sort_values(
            list(STUDENT_KEY)
        )
        .reset_index(drop=True)
    )

    features[
        "avg_clicks_per_active_day_30"
    ] = np.divide(
        features["total_clicks_30"],
        features["active_days_30"],
        out=np.zeros(
            len(features),
            dtype=float,
        ),
        where=features[
            "active_days_30"
        ].to_numpy() > 0,
    )

    return features


def build_early_assessment_features(
    assessments: pd.DataFrame,
    student_assessments: pd.DataFrame,
    module: str,
    presentation: str,
    cutoff_day: int,
) -> pd.DataFrame:
    """Aggregate assessment submissions available by the observation cutoff."""

    _validate_cutoff_day(cutoff_day)

    require_columns(
        assessments,
        (
            "code_module",
            "code_presentation",
            "id_assessment",
            "date",
        ),
        dataframe_name="assessments",
    )

    require_columns(
        student_assessments,
        (
            "id_assessment",
            "id_student",
            "date_submitted",
            "score",
        ),
        dataframe_name="student_assessments",
    )

    selected_assessments = select_presentation(
        assessments,
        module,
        presentation,
        dataframe_name="assessments",
    ).rename(columns={"date": "assessment_due_date"})

    # Work with a copy so the original DataFrame is not modified.
    submissions_source = student_assessments.copy()

    missing_markers = {
        "": pd.NA,
        "?": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "null": pd.NA,
        "None": pd.NA,
    }

    # Clean the assessment due date.
    due_date_values = (
        selected_assessments["assessment_due_date"]
        .astype("string")
        .str.strip()
        .replace(missing_markers)
    )

    selected_assessments["assessment_due_date"] = pd.to_numeric(
        due_date_values,
        errors="raise",
    )

    # Clean submission date and score.
    for column in ["date_submitted", "score"]:
        cleaned_values = (
            submissions_source[column]
            .astype("string")
            .str.strip()
            .replace(missing_markers)
        )

        submissions_source[column] = pd.to_numeric(
            cleaned_values,
            errors="raise",
        )

    assessment_lookup = selected_assessments[
        [
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_due_date",
        ]
    ].copy()

    submissions = submissions_source.merge(
        assessment_lookup,
        on="id_assessment",
        how="inner",
        validate="many_to_one",
    )

    # Keep only submissions available between day 0 and the cutoff day.
    early = submissions.loc[
        submissions["date_submitted"].between(
            0,
            cutoff_day,
            inclusive="both",
        )
    ].copy()

    if early.empty:
        return _empty_assessment_features()

    # Both columns are now numeric, so this comparison is valid.
    early["is_late"] = (
        early["assessment_due_date"].notna()
        & early["date_submitted"].gt(
            early["assessment_due_date"]
        )
    ).astype("int8")

    features = (
        early.groupby(list(STUDENT_KEY), as_index=False)
        .agg(
            assessments_submitted_30=(
                "id_assessment",
                "nunique",
            ),
            avg_score_30=(
                "score",
                "mean",
            ),
            late_submissions_30=(
                "is_late",
                "sum",
            ),
        )
        .sort_values(list(STUDENT_KEY))
        .reset_index(drop=True)
    )

    features["has_early_assessment"] = 1

    return features

def select_eligible_students(
    student_info: pd.DataFrame,
    student_registration: pd.DataFrame,
    module: str,
    presentation: str,
    cutoff_day: int,
) -> pd.DataFrame:
    """Select students whose outcome is not already known by the cutoff day.

    Students are retained when they are registered by the cutoff and have not
    withdrawn on or before it. Students who withdraw after the cutoff remain
    eligible because their later outcome is what the model attempts to flag.
    """

    _validate_cutoff_day(cutoff_day)

    selected_info = select_presentation(
        student_info,
        module,
        presentation,
        dataframe_name="student_info",
    )

    selected_registration = select_presentation(
        student_registration,
        module,
        presentation,
        dataframe_name="student_registration",
    )

    validate_unique_key(
        selected_info,
        STUDENT_KEY,
        dataframe_name="selected student_info",
    )

    validate_unique_key(
        selected_registration,
        STUDENT_KEY,
        dataframe_name="selected student_registration",
    )

    # CSV columns can sometimes be loaded as strings.
    # Convert them to numeric course-day values before comparing them
    # with the integer cutoff_day.
    date_columns = ["date_registration", "date_unregistration"]

    for column in date_columns:
        cleaned_values = (
            selected_registration[column]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "": pd.NA,
                    "?": pd.NA,
                    "NA": pd.NA,
                    "N/A": pd.NA,
                    "null": pd.NA,
                    "None": pd.NA,
                }
            )
        )

        selected_registration[column] = pd.to_numeric(
            cleaned_values,
            errors="raise",
        )

    registered_by_cutoff = (
        selected_registration["date_registration"].isna()
        | selected_registration["date_registration"].le(cutoff_day)
    )

    not_withdrawn_by_cutoff = (
        selected_registration["date_unregistration"].isna()
        | selected_registration["date_unregistration"].gt(cutoff_day)
    )

    eligible_registration = selected_registration.loc[
        registered_by_cutoff & not_withdrawn_by_cutoff,
        [*STUDENT_KEY, "date_registration"],
    ].copy()

    eligible_students = selected_info.merge(
        eligible_registration,
        on=list(STUDENT_KEY),
        how="inner",
        validate="one_to_one",
    )

    if eligible_students.empty:
        raise ValueError(
            "No eligible students remain after applying the registration and "
            "withdrawal cutoff rules."
        )

    return eligible_students

def build_student_feature_table(
    eligible_students: pd.DataFrame,
    vle_features: pd.DataFrame,
    assessment_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine early activity features into one modelling row per student."""

    require_columns(
        eligible_students,
        (
            *STUDENT_KEY,
            "final_result",
        ),
        dataframe_name=(
            "eligible_students"
        ),
    )

    validate_unique_key(
        eligible_students,
        STUDENT_KEY,
        dataframe_name=(
            "eligible_students"
        ),
    )

    model_data = (
        eligible_students
        .merge(
            vle_features,
            on=list(STUDENT_KEY),
            how="left",
            validate="one_to_one",
        )
        .merge(
            assessment_features,
            on=list(STUDENT_KEY),
            how="left",
            validate="one_to_one",
        )
    )

    zero_fill_columns = [
        "total_clicks_30",
        "active_days_30",
        "unique_resources_30",
        "avg_clicks_per_active_day_30",
        "assessments_submitted_30",
        "late_submissions_30",
        "has_early_assessment",
    ]

    for column in zero_fill_columns:
        model_data[column] = (
            pd.to_numeric(
                model_data[column],
                errors="coerce",
            )
            .fillna(0)
        )

    integer_columns = [
        "total_clicks_30",
        "active_days_30",
        "unique_resources_30",
        "assessments_submitted_30",
        "late_submissions_30",
        "has_early_assessment",
    ]

    model_data[
        integer_columns
    ] = model_data[
        integer_columns
    ].astype("int64")

    model_data[
        "support_needed"
    ] = (
        model_data[
            "final_result"
        ]
        .isin(
            [
                "Fail",
                "Withdrawn",
            ]
        )
        .astype("int8")
    )

    validate_unique_key(
        model_data,
        STUDENT_KEY,
        dataframe_name="model_data",
    )

    if not model_data[
        "support_needed"
    ].isin([0, 1]).all():
        raise ValueError(
            "support_needed must contain only 0 and 1."
        )

    if (
        model_data[
            "total_clicks_30"
        ] < 0
    ).any():
        raise ValueError(
            "total_clicks_30 cannot contain negative values."
        )

    return (
        model_data
        .sort_values(
            list(STUDENT_KEY)
        )
        .reset_index(drop=True)
    )