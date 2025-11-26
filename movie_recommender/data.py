from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Default dataset path relative to repo root
_DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "movies.csv"


def load_movies(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load and preprocess the movie dataset.

    Parameters
    ----------
    csv_path:
        Explicit path to the CSV file. When omitted, the bundled dataset in
        ``data/movies.csv`` is used.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset with ``title`` and ``genres`` columns prepared for
        modelling.
    """

    path = Path(csv_path) if csv_path else _DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Movie dataset not found at: {path}")

    df = pd.read_csv(path)
    df = _normalise_columns(df)
    df = _normalise_columns(df)
    
    # Fill text columns
    for col in ["genres", "keywords", "overview"]:
        df[col] = df[col].fillna("").astype(str)
    
    # Clean genres (pipes to spaces)
    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    
    # Ensure numeric/date columns are valid
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year.fillna(0).astype(int)
    
    df["title"] = df["title"].astype(str)
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist, remapping case differences if needed."""
    required_cols = {"title", "genres", "keywords", "overview", "vote_average", "release_date"}
    lower_map = {col.lower(): col for col in df.columns}
    rename_map = {}
    for required in required_cols:
        if required not in df.columns and required in lower_map:
            rename_map[lower_map[required]] = required

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


__all__ = ["load_movies"]
