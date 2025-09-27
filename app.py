from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from movie_recommender import build_default_recommender

app = Flask(__name__)
recommender = build_default_recommender()


@app.route("/")
def index():
    """Render the interactive demo page."""
    return render_template("index.html")


@app.route("/api/recommend")
def api_recommend():
    """Return JSON recommendations for the requested title."""
    title = request.args.get("title", "").strip()
    limit_raw = request.args.get("limit", "5")

    try:
        limit = max(1, min(20, int(limit_raw)))
    except ValueError:
        return jsonify({"error": "limit must be an integer between 1 and 20"}), 400

    if not title:
        return jsonify({"error": "title query parameter is required"}), 400

    recommendations = recommender.recommend(title, top_n=limit)
    if not recommendations:
        suggestions = recommender.suggest_titles(title, limit=5)
        return jsonify({"recommendations": [], "suggestions": suggestions})

    payload = {"recommendations": [rec.to_dict() for rec in recommendations]}
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True)
