# 🧠 Comparative Sentiment Analysis: Classical ML vs LSTM

This project performs sentiment analysis on IMDb movie reviews using both classical machine learning models and a deep learning model (LSTM). The goal is to classify reviews as positive or negative and compare the performance of traditional approaches vs neural networks.

---

## 📁 Project Overview

**Objective**:  
Predict sentiment polarity of movie reviews using:
- Naive Bayes
- Logistic Regression
- LSTM (Recurrent Neural Network)

**Dataset**:  
IMDb Dataset (preprocessed CSV)  
- 50,000 labeled reviews (balanced: 25K positive, 25K negative)

---

## 🧰 Tech Stack

| Task                 | Tools/Libraries Used                          |
|----------------------|-----------------------------------------------|
| Programming Language | Python                                        |
| Data Handling        | Pandas, NumPy                                 |
| NLP Preprocessing    | NLTK, Regular Expressions                     |
| ML Models            | Scikit-learn (Naive Bayes, Logistic Regression) |
| Deep Learning Model  | TensorFlow (Embedding → LSTM → Dense)         |
| Evaluation           | Precision, Recall, F1-score, Confusion Matrix |
| Visualization        | Matplotlib, Seaborn, TensorBoard              |

---

## 🧹 Preprocessing Pipeline

1. Convert text to lowercase  
2. Remove punctuation and numbers  
3. Remove stopwords using NLTK  
4. Tokenization and padding (for deep learning)  
5. TF-IDF vectorization (for classical models)  

---

## 🔍 Model Architectures

### 🔸 Naive Bayes & Logistic Regression
- Input vectorized using TF-IDF
- Trained using scikit-learn

### 🔹 LSTM Deep Learning Model
```python
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=500),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])


- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy
- Tracked training with TensorBoard


