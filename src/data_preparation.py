"""Data loading and validation utilities for the OULAD project."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd


REQUIRED_RAW_FILES: tuple[str, ...] = (
    "courses.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
)

STUDENT_KEY: tuple[str, ...] = (
    "code_module",
    "code_presentation",
    "id_student",
)


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root from a notebook, script, or test directory.

    The root is identified by the presence of both ``src`` and ``notebooks``.
    A clear error is raised instead of silently constructing incorrect paths.
    """

    current = (start or Path.cwd()).resolve()
    candidates = (current, *current.parents)

    for candidate in candidates:
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. Open VS Code from the "
        "eduinsight-student-success directory and run the notebook again."
    )


def validate_raw_data_files(
    raw_data_dir: Path,
    required_files: Iterable[str] = REQUIRED_RAW_FILES,
) -> None:
    """Raise a helpful error when one or more required CSV files are missing."""

    raw_data_dir = Path(raw_data_dir)
    missing = [
        name
        for name in required_files
        if not (raw_data_dir / name).is_file()
    ]

    if missing:
        formatted = "\n".join(f"- {name}" for name in missing)

        raise FileNotFoundError(
            f"Missing required OULAD files in {raw_data_dir}:\n"
            f"{formatted}\n"
            "Download and extract the dataset into data/raw before continuing."
        )


def load_csv(
    raw_data_dir: Path,
    filename: str,
    *,
    usecols: Sequence[str] | None = None,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load one CSV file with path validation and a consistent error message."""

    file_path = Path(raw_data_dir) / filename

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    return pd.read_csv(
        file_path,
        usecols=usecols,
        dtype=dtype,
        na_values=["?"],
        keep_default_na=True
    )


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    dataframe_name: str,
) -> None:
    """Validate that a DataFrame contains all columns required by an operation."""

    missing = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{', '.join(missing)}"
        )


def validate_unique_key(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    dataframe_name: str,
) -> None:
    """Ensure that a logical key identifies at most one row."""

    require_columns(
        dataframe,
        key_columns,
        dataframe_name=dataframe_name,
    )

    duplicate_mask = dataframe.duplicated(
        subset=list(key_columns),
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise ValueError(
            f"{dataframe_name} contains {duplicate_count} rows with duplicate "
            f"keys for {list(key_columns)}. Review the selected presentation "
            "before merging tables."
        )


def select_presentation(
    dataframe: pd.DataFrame,
    module: str,
    presentation: str,
    *,
    dataframe_name: str = "dataframe",
) -> pd.DataFrame:
    """Return a defensive copy for one module presentation."""

    require_columns(
        dataframe,
        ("code_module", "code_presentation"),
        dataframe_name=dataframe_name,
    )

    selected = dataframe.loc[
        dataframe["code_module"].eq(module)
        & dataframe["code_presentation"].eq(presentation)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No rows in {dataframe_name} match "
            f"module={module!r} and presentation={presentation!r}."
        )

    return selected


def summarize_missing_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return missing-value counts and percentages for every column."""

    missing_count = dataframe.isna().sum()

    summary = pd.DataFrame(
        {
            "missing_values": missing_count,
            "missing_percentage": (
                missing_count / len(dataframe) * 100
            ).round(2),
        }
    )

    return summary.sort_values(
        by=["missing_values", "missing_percentage"],
        ascending=False,
    )