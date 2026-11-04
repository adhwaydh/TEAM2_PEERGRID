"""
PeerGrid KEAM recommendation inference.

Run from the ml_backend folder:
    python predict.py --rank 12500 --category SM --course "Computer Science and Engineering"

This version is tolerant of datasets that do not contain college_code.
"""

from pathlib import Path
import argparse
import json
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "keam_recommender.pkl"
DATA_PATH = BASE_DIR / "dataset" / "keam_cutoff.csv"
META_PATH = BASE_DIR / "model" / "model_metadata.json"


def clean(value):
    return " ".join(str(value).split()).strip()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Make sure model/keam_recommender.pkl exists."
        )
    return joblib.load(MODEL_PATH)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Make sure dataset/keam_cutoff.csv exists."
        )

    data = pd.read_csv(DATA_PATH)

    required = [
        "year",
        "phase",
        "college",
        "college_type",
        "course",
        "category",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing required columns: "
            + ", ".join(missing)
        )

    for col in [
        "phase",
        "college",
        "college_type",
        "course",
        "category",
    ]:
        data[col] = data[col].astype(str).map(clean)

    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data.dropna(subset=["year"]).copy()
    data["year"] = data["year"].astype(int)

    return data


def recommend(
    rank: int,
    category: str,
    course: str,
    year: int | None = None,
    phase: str | None = None,
    top_n: int = 10,
):
    model = load_model()
    data = load_data()

    if year is None:
        if META_PATH.exists():
            with open(META_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            year = int(metadata.get("default_prediction_year", data["year"].max()))
        else:
            year = int(data["year"].max())

    category = clean(category)
    course = clean(course)

    candidates = data[
        (data["category"] == category)
        & (data["course"] == course)
    ].copy()

    if phase:
        phase = clean(phase)
        phase_candidates = candidates[candidates["phase"] == phase]
        if not phase_candidates.empty:
            candidates = phase_candidates

    if candidates.empty:
        return {
            "rank": int(rank),
            "category": category,
            "course": course,
            "prediction_year": int(year),
            "recommendations": [],
            "message": (
                "No historical records were found for this exact "
                "category/course combination."
            ),
        }

    # college_code is optional because some exported/renamed CSV files
    # may not contain it.
    has_college_code = "college_code" in candidates.columns

    candidate_cols = [
        "college",
        "college_type",
        "course",
        "category",
        "phase",
    ]
    if has_college_code:
        candidate_cols.insert(1, "college_code")

    candidates = candidates[candidate_cols].drop_duplicates()

    rows = []
    for _, row in candidates.iterrows():
        item = {
            "year": int(year),
            "phase": row["phase"],
            "college_type": row["college_type"],
            "course": row["course"],
            "category": row["category"],
            "college": row["college"],
        }

        if has_college_code:
            item["college_code"] = row["college_code"]

        rows.append(item)

    X = pd.DataFrame(rows)

    # The trained model only uses these six feature columns.
    model_features = [
        "year",
        "phase",
        "college_type",
        "course",
        "category",
        "college",
    ]

    predicted = model.predict(X[model_features])

    results = X.copy()
    results["predicted_closing_rank"] = (
        pd.Series(predicted).round().astype(int).values
    )

    # A positive margin means the student's rank is numerically better
    # (smaller) than the predicted closing rank.
    results["rank_margin"] = (
        results["predicted_closing_rank"] - int(rank)
    )

    def classify(margin):
        # Match labels are based on the distance from the student's
        # rank, not simply on whether the closing rank is extremely
        # large.
        if margin < -3000:
            return "Low Match"
        if margin < 0:
            return "Borderline"
        if margin <= 500:
            return "Strong Match"
        if margin <= 2000:
            return "Good Match"
        return "Possible Match"

    results["match"] = results["rank_margin"].map(classify)

    # ---------------------------------------------------------
    # RECOMMENDATION ORDER
    # ---------------------------------------------------------
    # A student's KEAM rank is better when it is SMALLER.
    #
    # Example:
    #   student rank = 12
    #   closing ranks = 15, 27, 500, 55928
    #
    # The closest realistic options should be shown first:
    #   15 -> 27 -> 500 -> 55928
    #
    # The old code sorted by the LARGEST positive margin first.
    # That caused a rank of 12 to prefer a closing rank such as
    # 55928, because 55928 - 12 is a very large positive number.
    #
    # We instead:
    #   1. Put realistic colleges (closing rank >= student rank)
    #      first.
    #   2. Sort those by the smallest rank difference.
    #   3. If we need more results, add "reach" colleges whose
    #      predicted closing rank is below the student's rank,
    #      again by closest difference.
    #
    # This makes the recommendation rank-aware.

    results["rank_distance"] = (
        results["predicted_closing_rank"] - int(rank)
    ).abs()

    realistic = results[results["rank_margin"] >= 0].copy()
    reach = results[results["rank_margin"] < 0].copy()

    realistic = realistic.sort_values(
        by=["rank_margin", "predicted_closing_rank"],
        ascending=[True, True],
    )

    reach = reach.sort_values(
        by=["rank_distance", "predicted_closing_rank"],
        ascending=[True, False],
    )

    results = pd.concat(
        [realistic, reach],
        ignore_index=True,
    )

    output = []

    for _, row in results.head(top_n).iterrows():
        item = {
            "college": row["college"],
            "college_type": row["college_type"],
            "course": row["course"],
            "category": row["category"],
            "phase": row["phase"],
            "predicted_closing_rank": int(row["predicted_closing_rank"]),
            "rank_margin": int(row["rank_margin"]),
            "match": row["match"],
        }

        if has_college_code and "college_code" in row.index:
            item["college_code"] = row["college_code"]

        output.append(item)

    return {
        "rank": int(rank),
        "category": category,
        "course": course,
        "prediction_year": int(year),
        "recommendations": output,
        "warning": (
            "These are historical-model recommendations, not an official "
            "KEAM allotment or admission guarantee."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--course", required=True)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--top-n", type=int, default=10)

    args = parser.parse_args()

    result = recommend(
        rank=args.rank,
        category=args.category,
        course=args.course,
        year=args.year,
        phase=args.phase,
        top_n=args.top_n,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
