# Patient Condition Classification Using Drug Reviews

An applied NLP system that classifies patient drug reviews into **Depression**, **Diabetes, Type 2**, or **High Blood Pressure** using an interpretable **TF-IDF + Linear SVM** pipeline.

This project demonstrates end-to-end machine learning, NLP preprocessing, model evaluation, Streamlit deployment, and product-focused AI application design.

> Portfolio theme: **Building intelligent AI systems that solve real-world problems.**

## Quick Links

- Portfolio: [portfolio-self-one-10.vercel.app](https://portfolio-self-one-10.vercel.app)
- Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Project review: [REVIEW.md](REVIEW.md)
- Main app file: [streamlit_app.py](streamlit_app.py)
- Notebook: [notebook.ipynb](notebook.ipynb)

## Highlights

- Two-page Streamlit application:
  - **AI Prediction System** for live review classification
  - **Technical Case Study** for the engineering journey
- Classical NLP pipeline using TF-IDF unigram and bigram features
- Tuned Linear SVM final model
- VADER sentiment analysis as supporting context
- Cached artifact loading and deployment-safe file paths
- Input validation, sample reviews, prediction strength signal, and medical disclaimer
- GitHub-ready documentation and deployment files

## Problem Statement

Patient drug reviews contain valuable signals about treatment experience, symptoms, side effects, and satisfaction. The goal of this project is to convert unstructured review text into a supervised classification system that predicts one of three selected condition categories.

This is an educational portfolio project and is **not medical advice**.

## Dataset

Source file: `dataset.xlsx`

The raw dataset contains **161,297** drug review records.

| Field | Description |
| --- | --- |
| `drugName` | Name of the reviewed drug |
| `condition` | Medical condition associated with the review |
| `review` | Patient-written drug review text |
| `rating` | Patient satisfaction rating from 1 to 10 |
| `date` | Review date |
| `usefulCount` | Number of users who found the review useful |

The modeling workflow focuses on three target classes:

| Condition | Reviews |
| --- | ---: |
| Depression | 9,069 |
| Diabetes, Type 2 | 2,554 |
| High Blood Pressure | 2,321 |

Final filtered subset: **13,944 reviews**.

## Machine Learning Pipeline

```text
Raw drug review
      ↓
Text cleaning
      ↓
Stopword removal + lemmatization
      ↓
TF-IDF vectorization
      ↓
Linear SVM classifier
      ↓
Condition prediction
```

## Modeling Approach

The project compares multiple classical ML baselines:

| Model | Macro F1 | Accuracy |
| --- | ---: | ---: |
| Linear SVM | 0.946 | 0.962 |
| SGD Classifier | 0.942 | 0.958 |
| Logistic Regression | 0.936 | 0.953 |
| Naive Bayes | 0.919 | 0.941 |

Final model:

- Model: **Linear SVM**
- Best `C`: **1**
- Test accuracy: **96.16%**
- Test macro F1: **94.60%**

## Streamlit Application

The app is designed as a polished AI product rather than a notebook demo.

### Page 1: AI Prediction System

- One-click sample reviews
- Patient review input
- Condition prediction
- Sentiment analysis
- Match strength indicator
- Responsible-use disclaimer
- GitHub and technical case study navigation

### Page 2: Technical Case Study

- Business problem
- Dataset overview
- EDA charts
- NLP preprocessing pipeline
- Model comparison
- Hyperparameter tuning
- Classification report
- Confusion matrix
- Deployment architecture
- Challenges, lessons learned, and future improvements

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

## Tech Stack

- Python
- pandas
- scikit-learn
- NLTK
- TF-IDF
- Linear SVM
- Streamlit
- Graphviz
- Jupyter Notebook

## Run Locally

```bash
git clone https://github.com/Pranav427/medical-condition-classification-drug-reviews-nlp.git
cd medical-condition-classification-drug-reviews-nlp
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment

Recommended platform: **Streamlit Community Cloud**

Main file:

```text
streamlit_app.py
```

Required runtime files:

- `requirements.txt`
- `runtime.txt`
- `model.pkl`
- `vectorizer.pkl`
- `label_encoder.pkl`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Limitations

- The model only predicts among three supported classes.
- Match strength is based on Linear SVM margin and is not a calibrated probability.
- The system is trained on historical drug review data and may not generalize to all medical contexts.
- This project is for educational and portfolio purposes only.

## Future Improvements

- Convert the notebook workflow into a reproducible `train.py` script.
- Save preprocessing, vectorization, and model inference as a single scikit-learn `Pipeline`.
- Add unit tests for preprocessing and prediction.
- Add a model card and dataset ethics note.
- Add calibrated probabilities only after validation.
- Add a short demo GIF and screenshots to the README.

## Portfolio Value

This project demonstrates practical NLP, supervised machine learning, model comparison, evaluation, artifact serialization, deployment readiness, and product-oriented AI application design.
