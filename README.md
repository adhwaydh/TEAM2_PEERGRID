# PeerGrid KEAM ML Backend

This folder contains the first version of the PeerGrid KEAM college
recommendation backend.

## 1. Install Python

Use Python 3.10+.

## 2. Create a virtual environment

Windows:
    python -m venv .venv
    .venv\Scripts\activate

macOS/Linux:
    python3 -m venv .venv
    source .venv/bin/activate

## 3. Install packages

    pip install -r requirements.txt

## 4. Train the model

    python train_model.py

This creates:

    model/keam_recommender.pkl
    model/model_metadata.json

## 5. Test a recommendation

Example:

    python predict.py --rank 12500 --category SM --course "Computer Science and Engineering"

The exact course and category must exist in the dataset.

## 6. Start the API

    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open:

    http://127.0.0.1:8000/docs

The Flutter application will eventually call:

    POST /predict

## Important

The current model is based on historical KEAM closing ranks from the supplied
2022-2026 dataset. It is a recommendation aid, not an official KEAM admission
prediction or guarantee.

The category codes in the dataset (for example SM, EZ, MU, SC, ST) must be
mapped carefully to the labels used in the Flutter UI before the final app
integration.
