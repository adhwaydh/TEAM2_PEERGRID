"""
PeerGrid KEAM College Recommendation Model
------------------------------------------
Train a historical KEAM closing-rank model.

The model predicts a college/course/category closing rank from:
    year, phase, college_type, course, category, college

The recommender then compares a student's KEAM rank with predicted
closing ranks and returns the closest matching colleges.

IMPORTANT:
This is a recommendation model based on historical closing ranks.
It is NOT an official KEAM admission predictor and does not guarantee admission.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "keam_cutoff.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "keam_recommender.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"


def clean_text(series):
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in ["phase", "college_code", "college", "college_type", "course", "category"]:
        df[col] = clean_text(df[col])

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")

    df = df.dropna(
        subset=[
            "year", "closing_rank", "college",
            "course", "category", "phase", "college_type"
        ]
    ).copy()

    df["year"] = df["year"].astype(int)
    df["closing_rank"] = df["closing_rank"].astype(int)
    df = df[df["closing_rank"] > 0].copy()

    return df


def build_pipeline():
    numeric_features = ["year"]
    categorical_features = [
        "phase",
        "college_type",
        "course",
        "category",
        "college",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=24,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    df = load_data()

    features = [
        "year",
        "phase",
        "college_type",
        "course",
        "category",
        "college",
    ]
    target = "closing_rank"

    X = df[features]
    y = df[target]

    # Hold out 20% for a basic validation check.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    # Retrain on all historical rows for the final saved model.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    joblib.dump(final_pipeline, MODEL_PATH)

    metadata = {
        "model_type": "RandomForestRegressor",
        "features": features,
        "target": target,
        "rows": int(len(df)),
        "years": sorted(df["year"].unique().tolist()),
        "courses": int(df["course"].nunique()),
        "colleges": int(df["college"].nunique()),
        "categories": sorted(df["category"].unique().tolist()),
        "validation_mae_rank": round(mae, 2),
        "validation_rmse_rank": round(rmse, 2),
        "latest_year": int(df["year"].max()),
        "default_prediction_year": int(df["year"].max()),
        "warning": (
            "Historical recommendation only. A recommendation is not "
            "an official admission guarantee."
        ),
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print("PeerGrid KEAM recommendation model trained successfully")
    print("=" * 60)
    print(f"Rows used:        {len(df):,}")
    print(f"Years:            {metadata['years']}")
    print(f"Colleges:         {metadata['colleges']:,}")
    print(f"Courses:          {metadata['courses']:,}")
    print(f"Categories:       {len(metadata['categories'])}")
    print(f"Validation MAE:   {mae:,.2f} rank positions")
    print(f"Validation RMSE:  {rmse:,.2f} rank positions")
    print(f"Saved model:      {MODEL_PATH}")
    print(f"Saved metadata:   {META_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
