from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_movies


@dataclass(frozen=True)
class Recommendation:
    """Represents a single recommended movie."""

    title: str
    score: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


class MovieRecommender:
    """Content-based movie recommender using genre TF-IDF similarity."""

    def __init__(self, movies: pd.DataFrame):
        if "title" not in movies or "genres" not in movies:
            raise ValueError("movies dataframe must contain 'title' and 'genres' columns")

        # Work on a copy with clean index to align matrices and rows.
        self._movies = movies.reset_index(drop=True).copy()
        self._vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._vectorizer.fit_transform(self._movies["genres"])

    @property
    def movies(self) -> pd.DataFrame:
        """Return the underlying movie metadata."""
        return self._movies.copy()

    def recommend(self, title: str, *, top_n: int = 5) -> List[Recommendation]:
        """Return similar movies for the provided title.

        Parameters
        ----------
        title:
            Movie title to search for. Supports exact match first, then
            case-insensitive substring fallback.
        top_n:
            Maximum number of recommendations to return.
        """
        if not title or not isinstance(title, str):
            raise ValueError("Provide a non-empty movie title string")

        idx = self._match_title(title)
        if idx is None:
            return []

        similarity_vector = cosine_similarity(self._tfidf_matrix[idx], self._tfidf_matrix).flatten()
        ranked_indices = similarity_vector.argsort()[::-1]
        ranked_indices = [i for i in ranked_indices if i != idx][:top_n]

        recommendations = [
            Recommendation(title=self._movies.iloc[i]["title"], score=float(similarity_vector[i]))
            for i in ranked_indices
        ]
        return recommendations

    def suggest_titles(self, query: str, *, limit: int = 10) -> list[str]:
        """Return a list of titles containing the query (case-insensitive)."""
        if not query:
            return []
        mask = self._movies["title"].str.contains(re.escape(query), case=False, na=False)
        return self._movies.loc[mask, "title"].head(limit).tolist()

    def _match_title(self, title: str) -> Optional[int]:
        # Try exact match first (casefold handles unicode case rules)
        exact = self._movies[self._movies["title"].str.casefold() == title.casefold()]
        if not exact.empty:
            return int(exact.index[0])

        # Fallback to substring search
        mask = self._movies["title"].str.contains(re.escape(title), case=False, na=False)
        fallback = self._movies[mask]
        if fallback.empty:
            return None
        return int(fallback.index[0])


def build_default_recommender(csv_path: Optional[str] = None) -> MovieRecommender:
    """Convenience helper returning a recommender with the bundled dataset."""
    movies = load_movies(csv_path)
    return MovieRecommender(movies)


__all__ = ["MovieRecommender", "Recommendation", "build_default_recommender"]
