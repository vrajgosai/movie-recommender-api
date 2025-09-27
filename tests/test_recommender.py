import pytest

from movie_recommender import build_default_recommender


@pytest.fixture(scope="session")
def recommender():
    return build_default_recommender()


def test_recommendations_for_known_title(recommender):
    results = recommender.recommend("Toy Story", top_n=5)
    assert results, "Expected at least one recommendation"
    assert len(results) <= 5

    titles = [rec.title for rec in results]
    assert "Toy Story" not in titles, "Recommendations should not include the queried movie"

    scores = [rec.score for rec in results]
    assert scores == sorted(scores, reverse=True)


def test_unknown_title_returns_empty_list(recommender):
    results = recommender.recommend("Completely Made Up Title", top_n=5)
    assert results == []


def test_suggest_titles_returns_partial_matches(recommender):
    suggestions = recommender.suggest_titles("story", limit=5)
    assert suggestions, "Expected to receive suggestion list"
    assert any("story" in title.lower() for title in suggestions)
