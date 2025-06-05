# 📊 Comparative Sentiment Analysis Using Classical and Deep Learning Models

This project performs sentiment analysis on movie reviews by comparing classical machine learning models (Naive Bayes, Logistic Regression) and a deep learning model (LSTM). The objective is to evaluate their effectiveness in classifying reviews as positive or negative.

---

## 🚀 Project Overview

🎯 **Goal**: Predict the sentiment of textual movie reviews using different ML approaches  
🧪 **Models Compared**:
- Naive Bayes (Multinomial)
- Logistic Regression
- Long Short-Term Memory (LSTM)

📊 **Dataset**: [IMDb Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
📁 Format: 50,000 labeled reviews (`positive` / `negative`)

---

## 🧰 Tech Stack

| Area                  | Tools/Libraries Used                                |
|-----------------------|-----------------------------------------------------|
| Programming Language  | Python                                              |
| Data Handling         | Pandas, NumPy                                       |
| Preprocessing         | NLTK, Regex, Scikit-learn, Keras Tokenizer          |
| Classical ML Models   | Scikit-learn (Naive Bayes, Logistic Regression)     |
| Deep Learning Model   | TensorFlow (LSTM with Embedding + LSTM + Dense)     |
| Evaluation            | Precision, Recall, F1-score, Confusion Matrix       |
| Visualization         | Matplotlib, Seaborn, TensorBoard                    |

---

## 🧹 Preprocessing Pipeline

1. Lowercasing
2. Removal of punctuation, numbers, and stopwords
3. Tokenization
4. Vectorization:
   - **TF-IDF** for classical models
   - **Embedding Layer** for LSTM model

---

## 🧠 Model Architectures

### 🧪 Classical Models
- **TF-IDF Vectorizer + Naive Bayes**
- **TF-IDF Vectorizer + Logistic Regression**

### 🔥 Deep Learning (LSTM)
```python
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=MAXLEN),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
