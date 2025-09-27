#!/usr/bin/env python3
"""Command line interface for the movie recommender."""

from __future__ import annotations

import argparse
from typing import Sequence

from movie_recommender import build_default_recommender


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend movies by genre similarity.")
    parser.add_argument("title", help="Movie title to search for.")
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=5,
        help="Number of recommendations to return (default: 5).",
    )
    parser.add_argument(
        "--csv",
        help="Optional path to a custom movies.csv file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    recommender = build_default_recommender(args.csv)
    recommendations = recommender.recommend(args.title, top_n=args.top_n)

    if not recommendations:
        print(f"No matches found for '{args.title}'.")
        suggestions = recommender.suggest_titles(args.title, limit=5)
        if suggestions:
            print("Did you mean:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
        return 1

    print(f"Recommendations for {args.title!r}:")
    for rec in recommendations:
        print(f"  - {rec.title} (score={rec.score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
