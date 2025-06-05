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
``` 

- **Optimizer**: Adam  
- **Loss**: Binary Crossentropy  
- **Metrics**: Accuracy  
- **Tracked training**: TensorBoard

## 📊 Evaluation Metrics

Each model was evaluated using:

- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **TensorBoard** (for LSTM)

---

## 📈 Results

| Model                | Precision | Recall | F1 Score |
|---------------------|-----------|--------|----------|
| Naive Bayes         | 0.8511    | 0.8531 | 0.8521   |
| Logistic Regression | 0.8782    | 0.9049 | 0.8914   |
| LSTM                | ~0.91     | ~0.89  | ~0.90    |

> 🔎 The LSTM model slightly outperforms classical models in F1 score due to its ability to learn deeper sequential context in reviews.

---

## 📂 Project Structure
.
├── Sentiment_Analysis.ipynb   # Jupyter notebook with all models
├── IMDB Dataset.csv           # Dataset used for analysis
├── README.md                  # Full project overview

## How to Run This Project

```bash
# 1. Clone the repository
git clone (https://github.com/srushtilohiya/Comparative-Sentiment-Analysis-Using-Classical-and-Deep-Learning-Models.git)
cd sentiment-analysis

# 2. Install required libraries
pip install pandas numpy nltk scikit-learn tensorflow matplotlib seaborn

# 3. Run the Jupyter notebook
jupyter notebook Sentiment_Analysis.ipynb





