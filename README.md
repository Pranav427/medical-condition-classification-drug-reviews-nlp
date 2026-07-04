# Patient Condition Classification Using Drug Reviews

An applied NLP project that classifies patient drug reviews into three medical condition categories: **Depression**, **Diabetes, Type 2**, and **High Blood Pressure**. The project uses a practical TF-IDF + Linear SVM pipeline and includes a Streamlit app for interactive prediction.

This project supports the portfolio theme: **building intelligent AI systems that solve real-world problems through Applied AI, Machine Learning, and intelligent software engineering.**

## Project Overview

Patient drug reviews contain useful signals about symptoms, treatment experience, side effects, and perceived effectiveness. This project turns those unstructured reviews into a supervised text classification workflow that predicts the likely condition category from review text.

The deployed app also includes a VADER sentiment signal to help interpret whether the review tone is positive, negative, or neutral. Sentiment is used as supporting context, not as the supervised target.

## Business Objective

The objective is to demonstrate how NLP can extract structure from patient-generated healthcare text. A lightweight classifier can help organize large volumes of reviews by condition and support exploratory analysis of patient experiences.

This application is for educational and portfolio use only. It is not medical advice and should not be used for diagnosis.

## Dataset

Source file: `dataset.xlsx`

The dataset contains 161,297 drug review records with these fields:

| Column | Description |
| --- | --- |
| `drugName` | Name of the reviewed drug |
| `condition` | Medical condition associated with the review |
| `review` | Patient-written drug review text |
| `rating` | Patient satisfaction rating from 1 to 10 |
| `date` | Review date |
| `usefulCount` | Number of users who found the review useful |

For this project, the workflow filters the data to:

| Condition | Records |
| --- | ---: |
| Depression | 9,069 |
| Diabetes, Type 2 | 2,554 |
| High Blood Pressure | 2,321 |

Final filtered dataset size: **13,944 reviews**.

## Methodology

1. Load and inspect the drug review dataset.
2. Filter to the three target conditions.
3. Clean review text by lowercasing, removing HTML/URLs/punctuation, removing stopwords, and lemmatizing tokens.
4. Encode condition labels with `LabelEncoder`.
5. Split the data with stratification to preserve class balance.
6. Convert text to TF-IDF features with up to 5,000 unigram and bigram features.
7. Compare multiple baseline models:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM
   - SGD Classifier
8. Tune Linear SVM with cross-validated grid search.
9. Save the trained model, vectorizer, and label encoder as `.pkl` artifacts.
10. Deploy the prediction pipeline with Streamlit.

## Results

The tuned Linear SVM model achieved the strongest performance in the notebook.

| Model | Macro F1 | Accuracy |
| --- | ---: | ---: |
| Linear SVM | 0.946 | 0.962 |
| SGD Classifier | 0.942 | 0.958 |
| Logistic Regression | 0.936 | 0.953 |
| Naive Bayes | 0.919 | 0.941 |

Final tuned Linear SVM:

- Best `C`: `1`
- Test accuracy: **0.9616**
- Test macro F1: **0.9460**

## Application

The Streamlit app:

- Provides a two-page experience: **AI Prediction System** and **Technical Case Study**.
- Loads the saved Linear SVM, TF-IDF vectorizer, and label encoder.
- Applies preprocessing consistent with the training notebook.
- Predicts the condition category.
- Shows VADER sentiment.
- Shows a margin-based prediction strength indicator.
- Handles empty and very short inputs.
- Uses deployment-safe paths and cached model loading.
- Includes one-click sample reviews for the supported classes.
- Provides concise navigation, product links, model context, and a visual engineering case study.
- Presents results with clear warnings when the prediction strength is low.
- Includes product-level disclaimers suitable for a healthcare-adjacent AI demo.

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project Structure

```text
.
├── streamlit_app.py
├── dataset.xlsx
├── model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── notebook.ipynb
├── project_brief.docx
├── README.md
├── DEPLOYMENT.md
├── REVIEW.md
├── requirements.txt
├── runtime.txt
└── .gitignore
```

## Technologies Used

- Python
- pandas
- scikit-learn
- NLTK
- TF-IDF
- Linear SVM
- Streamlit
- Jupyter Notebook

## Deployment

Recommended first deployment target: **Streamlit Community Cloud**.

Required files:

- `streamlit_app.py`
- `requirements.txt`
- `runtime.txt`
- `model.pkl`
- `vectorizer.pkl`
- `label_encoder.pkl`

See [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step deployment instructions.

## Suggested Screenshots

Add these images to the repository before public release:

- Streamlit app home screen
- Example prediction for a diabetes review
- Example prediction for a blood pressure review
- Confusion matrix from the notebook
- Model comparison table

## Future Improvements

- Convert the training workflow into a reproducible `train.py` script.
- Save the text preprocessor and classifier as one scikit-learn `Pipeline`.
- Add unit tests for preprocessing and inference.
- Add model cards and dataset ethics notes.
- Add calibrated probabilities only if probability calibration is validated.
- Add lightweight error analysis with representative misclassified examples.

## Portfolio Summary

This project demonstrates practical NLP, supervised machine learning, model evaluation, artifact serialization, and Streamlit deployment. It is intentionally built with a classical ML approach because TF-IDF + Linear SVM is appropriate, interpretable, efficient, and strong for this three-class text classification task.
