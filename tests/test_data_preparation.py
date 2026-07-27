from __future__ import annotations

import pandas as pd
import pytest

from src.data_preparation import (
    select_presentation,
    summarize_missing_values,
    validate_unique_key,
)


def test_select_presentation_returns_copy() -> None:
    source = pd.DataFrame(
        {
            "code_module": [
                "BBB",
                "CCC",
            ],
            "code_presentation": [
                "2013J",
                "2014J",
            ],
            "value": [
                1,
                2,
            ],
        }
    )

    selected = select_presentation(
        source,
        "BBB",
        "2013J",
    )

    selected.loc[
        :,
        "value",
    ] = 99

    assert (
        selected[
            "value"
        ].iloc[0]
        == 99
    )

    assert (
        source[
            "value"
        ].iloc[0]
        == 1
    )


def test_validate_unique_key_rejects_duplicates() -> None:
    duplicate_data = pd.DataFrame(
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
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_unique_key(
            duplicate_data,
            (
                "code_module",
                "code_presentation",
                "id_student",
            ),
            dataframe_name=(
                "duplicate_data"
            ),
        )


def test_missing_value_summary_is_sorted() -> None:
    data = pd.DataFrame(
        {
            "a": [
                1,
                None,
                None,
            ],
            "b": [
                1,
                2,
                None,
            ],
        }
    )

    summary = (
        summarize_missing_values(
            data
        )
    )

    assert (
        summary.index.tolist()
        == ["a", "b"]
    )

    assert (
        summary.loc[
            "a",
            "missing_values",
        ]
        == 2
    )