# Deployment Guide

## Recommended Target: Streamlit Community Cloud

This is the simplest deployment path for the current project because the application is already built with Streamlit and uses small serialized model artifacts.

## Files Required for Deployment

Keep these files in the GitHub repository root:

- `streamlit_app.py`
- `requirements.txt`
- `runtime.txt`
- `model.pkl`
- `vectorizer.pkl`
- `label_encoder.pkl`

The dataset and notebooks are useful for transparency, but the app does not need `dataset.xlsx` at runtime.

## Local Verification

From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local Streamlit URL and test:

- Empty input
- Very short input
- A diabetes-related review
- A depression-related review
- A high-blood-pressure-related review

## Streamlit Community Cloud Steps

1. Push the project to GitHub.
2. Go to Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file path to `streamlit_app.py`.
5. Deploy.
6. Confirm that the app loads all `.pkl` artifacts successfully.

## Hugging Face Spaces Alternative

1. Create a new Space.
2. Choose Streamlit as the SDK.
3. Upload the required files.
4. Keep `streamlit_app.py` at the Space root.
5. Let Hugging Face install dependencies from `requirements.txt`.

## Production Notes

- The app uses `pickle`, so only load model files that you created and trust.
- `scikit-learn==1.6.1` is pinned because the saved artifacts were created with that version.
- The current model supports only three classes: Depression, Diabetes Type 2, and High Blood Pressure.
- The prediction strength shown in the app is margin-based, not a calibrated probability.
- This project is educational and should not be presented as a diagnostic medical device.
