# Project Review

## Executive Summary

This is a strong applied NLP portfolio project with a clear objective: classify patient drug reviews into three condition categories using supervised machine learning. The model choice is sensible. TF-IDF + Linear SVM is fast, explainable, easy to deploy, and performs well on this dataset.

The project already demonstrates meaningful machine learning engineering skills: dataset filtering, text preprocessing, EDA, model comparison, hyperparameter tuning, evaluation, serialization, and Streamlit deployment. The main weaknesses were deployment polish, preprocessing mismatch in the app, missing dependency files, and limited public-facing documentation. Those have been addressed with an updated `streamlit_app.py`, `README.md`, `requirements.txt`, `runtime.txt`, `.gitignore`, and this review document.

## Understanding of the Project

The business problem is to organize patient-written drug reviews by likely medical condition. The target users are recruiters, reviewers, students, and technical evaluators who want to see a complete applied NLP workflow.

The dataset contains 161,297 raw drug reviews. The project focuses on three classes:

- Depression: 9,069 records
- Diabetes, Type 2: 2,554 records
- High Blood Pressure: 2,321 records

The ML pipeline cleans review text, removes stopwords, lemmatizes tokens, transforms text with TF-IDF, trains several classifiers, selects Linear SVM, tunes `C`, evaluates on a stratified test split, and saves artifacts for Streamlit inference.

## Improvements Implemented

### 1. Fixed Inference Preprocessing Mismatch

Issue: The notebook trained on cleaned text, but the original Streamlit app passed raw user input directly into the TF-IDF vectorizer.

Why it matters: Training-serving skew can reduce prediction quality because deployment inputs do not match the training distribution.

Solution: Added the same lowercasing, HTML cleanup, URL removal, punctuation removal, stopword removal, and lemmatization in `streamlit_app.py`.

Benefit: The deployed model now receives text in the same format used during training.

### 2. Added Robust Artifact Paths

Issue: The original app loaded `.pkl` files using relative paths, which can break if the app is launched from another working directory.

Solution: The updated app resolves model artifacts relative to `streamlit_app.py`.

Benefit: More reliable local and cloud deployment.

### 3. Added Cached Model and NLP Loading

Issue: Model loading and NLTK setup should not repeat unnecessarily during Streamlit reruns.

Solution: Added `st.cache_resource` for model artifacts and NLTK tools.

Benefit: Faster app startup and smoother interaction.

### 4. Fixed NLTK Runtime Handling

Issue: The previous try/except guarded only the import, but VADER can fail later when the lexicon is missing.

Solution: Added explicit NLTK resource checks and quiet downloads for stopwords, WordNet, and VADER.

Benefit: More reliable deployment in clean environments.

### 5. Added Input Validation

Issue: The app handled empty input but not very short or low-information text.

Solution: Added minimum input and post-cleaning validation.

Benefit: Better UX and fewer meaningless predictions.

### 6. Added Professional Documentation

Issue: The project lacked README, deployment instructions, dependency pinning, and repository hygiene files.

Solution: Added `README.md`, `DEPLOYMENT.md`, `REVIEW.md`, `requirements.txt`, `runtime.txt`, and `.gitignore`.

Benefit: The repository is now easier to run, review, deploy, and present publicly.

## Complete Issue List

### Critical

- App inference preprocessing did not match training preprocessing.
- No dependency file for reproducible installation.
- Pickled scikit-learn artifacts were created with version 1.6.1, so deployment should pin that version.
- App used relative artifact paths that can break in cloud deployment.

### High Priority

- Convert notebook training workflow into a reproducible script.
- Save preprocessing + vectorizer + model as one scikit-learn `Pipeline`.
- Add basic tests for preprocessing, artifact loading, and prediction.
- Keep `streamlit_app.py` as the deployment source of truth.
- Add a medical disclaimer to the README and app.

### Medium Priority

- Add confusion matrix image and model comparison screenshot to README.
- Add an error analysis section with examples of misclassifications.
- Improve notebook markdown and fix typos.
- Add a model card describing intended use, limitations, and risks.
- Add a small sample dataset if the full dataset is too large for GitHub.

### Nice to Have

- Add GitHub Actions for linting or smoke tests.
- Add a short demo GIF.
- Add a portfolio case-study page.
- Add calibrated probabilities only after validating calibration.

## Machine Learning Review

The dataset preparation is appropriate for a focused classification project. Filtering to three clinically distinct classes makes the problem clear and achievable. Stratified splitting is a good choice because the classes are imbalanced.

The model comparison is strong for a portfolio project. Linear SVM is a good final choice because it performs best and is a proven baseline for TF-IDF text classification.

The main ML improvement would be packaging preprocessing, vectorization, and classification into a single pipeline. This would prevent training-serving skew and simplify future retraining.

## NLP Review

The project uses standard NLP preprocessing: lowercasing, punctuation removal, stopword removal, and lemmatization. This is acceptable for a classical ML approach.

One caution: removing stopwords can remove negation words like "not", which may matter in medical reviews. A future version could preserve negation terms such as `not`, `no`, and `never`.

## Software Engineering Review

The original project was notebook-centered, which is normal for learning and exploration. For public release, the deployment code needs to be treated as production-facing source code. The updated app improves maintainability with functions, constants, cached loading, and explicit error handling.

The next maturity step is to create:

- `src/preprocessing.py`
- `src/predict.py`
- `train.py`
- `tests/test_preprocessing.py`
- `tests/test_prediction.py`

## UI/UX Review

The app is clean and focused. The updated version improves the UI by using a primary button, clearer labels, metrics, an expandable preprocessed-text view, and a visible educational disclaimer.

Avoid making the app flashy. Recruiters will value clarity, stable behavior, and a professional explanation more than decorative UI.

## Deployment Readiness

Deployment readiness is now much stronger because the required files are present and dependency versions are pinned. Streamlit Community Cloud is the best first platform.

Before publishing, verify the app in a fresh virtual environment.

## Recruiter Evaluation

Would this impress recruiters? Yes, if presented well. The project demonstrates practical NLP, model comparison, evaluation, and deployment. It is stronger than a notebook-only project because it includes a runnable app and deployment documentation.

Strongest aspects:

- Clear real-world NLP use case
- Strong classical ML baseline
- Good model performance
- Streamlit deployment
- Practical engineering improvements

What weakens it:

- Notebook is not yet converted into a reproducible training script
- No automated tests yet
- No model card or error analysis yet
- Dataset has medical context, so limitations must be stated carefully

## MSc Artificial Intelligence Review

This project supports an MSc AI application because it shows independent applied ML work, supervised learning, NLP preprocessing, evaluation, and deployment thinking. To strengthen the academic angle, add error analysis, discuss class imbalance, explain model choice, and include limitations around medical text classification.

## Interview Preparation

Likely interview questions:

- Why did you choose TF-IDF + Linear SVM instead of BERT?
- How did you handle class imbalance?
- What does macro F1 tell you here?
- How did you prevent data leakage?
- What are the limitations of VADER sentiment on medical reviews?
- How would you retrain and redeploy the model?
- Why is inference preprocessing important?
- What would change if the project had 50 condition classes?

30-second explanation:

> I built an NLP system that classifies patient drug reviews into Depression, Diabetes Type 2, or High Blood Pressure. I cleaned and explored a 161k-record drug review dataset, filtered it to three target conditions, compared several classical ML models using TF-IDF features, and selected a tuned Linear SVM because it achieved the best macro F1 and accuracy. I then serialized the model artifacts and deployed them in a Streamlit app with matching inference preprocessing, sentiment analysis, validation, and deployment documentation.

## GitHub Checklist

- Add a concise repository description.
- Add topics: `nlp`, `machine-learning`, `streamlit`, `scikit-learn`, `tfidf`, `svm`, `healthcare-ai`.
- Add screenshots in an `assets/` folder.
- Add a demo GIF.
- Add a license.
- Add a short model card.
- Keep the README focused and recruiter-friendly.

## Suggested Diagrams

Architecture diagram:

```text
User Review -> Text Preprocessing -> TF-IDF Vectorizer -> Linear SVM -> Condition Prediction
                                      |
                                      -> VADER Sentiment -> Sentiment Label
```

Workflow diagram:

```text
Dataset -> EDA -> Cleaning -> Train/Test Split -> TF-IDF -> Model Comparison -> Tuning -> Evaluation -> Streamlit App
```

## Final Scores

| Area | Score |
| --- | ---: |
| Business Understanding | 8/10 |
| Data Preparation | 8/10 |
| Feature Engineering | 8/10 |
| Machine Learning Pipeline | 8/10 |
| Model Performance | 9/10 |
| Software Engineering | 7/10 |
| Code Quality | 7/10 |
| UI/UX | 7/10 |
| Documentation | 8/10 |
| GitHub Quality | 8/10 |
| Deployment Readiness | 8/10 |
| Production Readiness | 7/10 |
| Portfolio Value | 8/10 |
| Recruiter Impression | 8/10 |
| MSc Application Value | 8/10 |
| Overall Project Quality | 8/10 |

## Remaining Best Next Step

The highest-value next improvement is to convert the notebook workflow into a reproducible training script and save a single scikit-learn pipeline artifact. That would make the project feel more like a maintained ML system than a notebook export.
