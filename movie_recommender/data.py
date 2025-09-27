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
    df["genres"] = df["genres"].fillna(" ").astype(str).str.replace("|", " ", regex=False)
    df["title"] = df["title"].astype(str)
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist, remapping case differences if needed."""
    required_cols = {"title", "genres"}
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
