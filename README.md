# 🎬 Movie Recommender (Flask + TF-IDF)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Flask](https://img.shields.io/badge/Flask-API-green)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20Cosine-orange)]()
[![License](https://img.shields.io/badge/License-MIT-black)]()

Content-based movie recommendations powered by TF-IDF over genres, with:
- A reusable Python package (`movie_recommender`)
- CLI tool for quick lookups (`python recommender.py "Toy Story"`)
- Flask JSON API + polished vanilla JS demo (`/`)
- Pytest coverage for core behaviours

---

## 🚀 Quickstart
```bash
# 1. Clone & enter the project
git clone https://github.com/<your-username>/movie-recommender-api.git
cd movie-recommender-api

# 2. Create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .[dev]        # registers the package + pytest extra

# 4. Run the Flask app
flask --app app run --reload
# visit http://127.0.0.1:5000/

# 5. (Optional) Serve the static page separately
python -m http.server 5500
# visit http://localhost:5500/templates/index.html
```

### 📂 Dataset
- The repo ships with a 5k-row sample (`data/movies.csv`) to keep GitHub-friendly (<100 MB) while providing rich demos.
- For the full MovieLens metadata, download from [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/), save as `data/movies_full.csv`, then replace the sample: `cp data/movies_full.csv data/movies.csv`.

### ✅ Tests
```bash
pytest
```

---

## 🧰 Project Structure
```
├── app.py                 # Flask entry point (API + HTML template)
├── data/
│   ├── movies.csv         # Sample dataset (5k rows)
│   └── movies_full.csv    # (optional) keep full dataset locally, ignored by git
├── movie_recommender/
│   ├── __init__.py
│   ├── data.py            # loading/cleaning helpers
│   └── recommender.py     # MovieRecommender + Recommendation dataclass
├── recommender.py         # CLI wrapper
├── templates/index.html   # Demo UI
├── tests/
│   ├── conftest.py
│   └── test_recommender.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 📄 License
MIT © 2024 Vraj Gosai
