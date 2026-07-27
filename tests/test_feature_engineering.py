from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import (
    build_early_assessment_features,
    build_early_vle_features,
    build_student_feature_table,
    select_eligible_students,
)


def test_vle_features_respect_presentation_and_cutoff(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "code_module": [
                "BBB",
                "BBB",
                "BBB",
                "CCC",
            ],
            "code_presentation": [
                "2013J",
                "2013J",
                "2013J",
                "2014J",
            ],
            "id_student": [
                1,
                1,
                1,
                1,
            ],
            "id_site": [
                10,
                10,
                11,
                10,
            ],
            "date": [
                1,
                2,
                40,
                1,
            ],
            "sum_click": [
                3,
                4,
                100,
                50,
            ],
        }
    )

    path = (
        tmp_path
        / "studentVle.csv"
    )

    source.to_csv(
        path,
        index=False,
    )

    result = (
        build_early_vle_features(
            path,
            module="BBB",
            presentation="2013J",
            cutoff_day=30,
            chunksize=2,
        )
    )

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "total_clicks_30",
        ]
        == 7
    )

    assert (
        result.loc[
            0,
            "active_days_30",
        ]
        == 2
    )

    assert (
        result.loc[
            0,
            "unique_resources_30",
        ]
        == 1
    )


def test_assessment_features_use_only_information_available_by_cutoff() -> None:
    assessments = pd.DataFrame(
        {
            "code_module": [
                "BBB",
                "BBB",
            ],
            "code_presentation": [
                "2013J",
                "2013J",
            ],
            "id_assessment": [
                100,
                101,
            ],
            "date": [
                10,
                50,
            ],
        }
    )

    submissions = pd.DataFrame(
        {
            "id_assessment": [
                100,
                101,
            ],
            "id_student": [
                1,
                1,
            ],
            "date_submitted": [
                12,
                40,
            ],
            "score": [
                70.0,
                90.0,
            ],
        }
    )

    result = (
        build_early_assessment_features(
            assessments,
            submissions,
            module="BBB",
            presentation="2013J",
            cutoff_day=30,
        )
    )

    assert (
        result.loc[
            0,
            "assessments_submitted_30",
        ]
        == 1
    )

    assert (
        result.loc[
            0,
            "avg_score_30",
        ]
        == 70.0
    )

    assert (
        result.loc[
            0,
            "late_submissions_30",
        ]
        == 1
    )


def test_eligibility_excludes_known_withdrawals_and_late_registration() -> None:
    student_info = pd.DataFrame(
        {
            "code_module": [
                "BBB",
                "BBB",
                "BBB",
            ],
            "code_presentation": [
                "2013J",
                "2013J",
                "2013J",
            ],
            "id_student": [
                1,
                2,
                3,
            ],
            "final_result": [
                "Pass",
                "Withdrawn",
                "Fail",
            ],
        }
    )

    registration = pd.DataFrame(
        {
            "code_module": [
                "BBB",
                "BBB",
                "BBB",
            ],
            "code_presentation": [
                "2013J",
                "2013J",
                "2013J",
            ],
            "id_student": [
                1,
                2,
                3,
            ],
            "date_registration": [
                -20,
                -10,
                40,
            ],
            "date_unregistration": [
                np.nan,
                20,
                np.nan,
            ],
        }
    )

    eligible = (
        select_eligible_students(
            student_info,
            registration,
            module="BBB",
            presentation="2013J",
            cutoff_day=30,
        )
    )

    assert (
        eligible[
            "id_student"
        ].tolist()
        == [1]
    )


def test_student_feature_table_zero_fills_absent_activity() -> None:
    eligible = pd.DataFrame(
        {
            "code_module": [
                "BBB",
                "BBB",
            ],
            "code_presentation": [
                "2013J",
                "2013J",
            ],
            "id_student": [
                1,
                2,
            ],
            "final_result": [
                "Pass",
                "Fail",
            ],
            "num_of_prev_attempts": [
                0,
                1,
            ],
            "studied_credits": [
                60,
                60,
            ],
            "date_registration": [
                -10,
                -5,
            ],
        }
    )

    vle = pd.DataFrame(
        {
            "code_module": [
                "BBB"
            ],
            "code_presentation": [
                "2013J"
            ],
            "id_student": [
                1
            ],
            "total_clicks_30": [
                20
            ],
            "active_days_30": [
                4
            ],
            "unique_resources_30": [
                3
            ],
            "avg_clicks_per_active_day_30": [
                5.0
            ],
        }
    )

    assessment = pd.DataFrame(
        columns=[
            "code_module",
            "code_presentation",
            "id_student",
            "assessments_submitted_30",
            "avg_score_30",
            "late_submissions_30",
            "has_early_assessment",
        ]
    )

    result = (
        build_student_feature_table(
            eligible,
            vle,
            assessment,
        )
    )

    second_student = (
        result.loc[
            result[
                "id_student"
            ].eq(2)
        ].iloc[0]
    )

    assert (
        second_student[
            "total_clicks_30"
        ]
        == 0
    )

    assert (
        second_student[
            "assessments_submitted_30"
        ]
        == 0
    )

    assert pd.isna(
        second_student[
            "avg_score_30"
        ]
    )

    assert (
        second_student[
            "support_needed"
        ]
        == 1
    )