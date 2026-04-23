# 🩺 Medical Condition Classification from Drug Reviews (Data Analytics & NLP)

## 📌 Project Overview
This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify a patient’s medical condition based on written drug reviews.

Using patient review text, the model predicts conditions such as:

- Depression
- High Blood Pressure
- Type 2 Diabetes

The project also performs **sentiment analysis** to identify whether a patient review reflects a positive, negative, or neutral experience.

---

## 🎯 Business Objective

Patient reviews contain valuable information about:

- Drug effectiveness
- Side effects
- Patient satisfaction
- Treatment experiences

This project analyzes those reviews and predicts patient conditions from text data, helping support:

- Healthcare insights
- Drug recommendation systems
- Review analytics
- Clinical decision support systems

---

## 🗂 Dataset Features

The dataset contains:

- Drug Name
- Condition
- Patient Review
- Rating
- Review Date
- Useful Count

Dataset Size:
- 161,000+ patient drug reviews

---

## 🧠 Project Workflow

### 1. Data Preprocessing
- Text cleaning
- Lowercasing
- Stopword removal
- Tokenization
- Lemmatization

---

### 2. Exploratory Data Analysis
- Condition distribution analysis
- Rating analysis
- Word frequency patterns
- Review sentiment patterns

---

### 3. Feature Engineering
- TF-IDF Vectorization
- Text feature extraction

---

### 4. Model Building
Machine Learning models tested:

- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

Final selected model:
✅ Tuned Linear SVM

---

## 📈 Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 💬 Sentiment Analysis
Using **NLTK VADER**, the project also detects:

- Positive Reviews 😊  
- Negative Reviews 😟  
- Neutral Reviews 😐  

---

## 🚀 Deployment

The project is deployed using **Streamlit**.

Users can:

1. Enter a patient drug review  
2. Predict the medical condition  
3. View review sentiment  

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF
- Streamlit

---

## 📂 Project Files

- NLP Project Notebook (`.ipynb`)
- Streamlit Application (`app.py`)
- Trained Model (`.pkl`)
- TF-IDF Vectorizer
- Label Encoder
- Dataset files

---

## 💡 Sample Input

Patient Review:
"This medicine helped control my blood pressure and I feel stable now."

Prediction:
- Condition: High Blood Pressure  
- Sentiment: Positive 😊

---

## 🚀 Skills Demonstrated

- Natural Language Processing (NLP)
- Text Classification
- Feature Engineering
- Machine Learning Modeling
- Sentiment Analysis
- Streamlit Deployment

---

## ⭐ Project Highlights

✔ End-to-End ML Project  
✔ NLP + Data Analytics  
✔ Model Deployment  
✔ Healthcare Use Case  
✔ Real-world Classification Problem

---

## ✅ Project Status

Completed and Deployed

---

## 🔮 Future Improvements

- Deep Learning using LSTM / BERT
- Drug recommendation engine
- Multi-condition classification
- Flask/FastAPI production deployment

---

