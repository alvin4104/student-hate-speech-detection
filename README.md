# Hate Speech Detection in Student Online Forums

**Project Type:** Text Classification (Hate Speech Detection)  
**Data Source:** Student online forum comments  
**Labels:** HATE / NON-HATE  

---

## Project Overview

This project builds a binary text classifier to automatically detect hate speech in student online forums. Comments are classified as **HATE** (offensive, discriminatory, or harmful language targeting students) or **NON-HATE** (constructive, respectful academic communication) using TF-IDF features and three supervised machine learning algorithms.

## Models Used

| Model | Type |
|-------|------|
| Naive Bayes | Machine Learning |
| Support Vector Machine (SVM) | Machine Learning |
| Logistic Regression | Machine Learning |

## Pipeline

```
Raw Student Forum Comments
      ↓
Text Preprocessing (lowercase, remove URLs/digits/punctuation, stop words)
      ↓
TF-IDF Vectorization (max 5,000 features, unigrams + bigrams)
      ↓
80/20 Stratified Train/Test Split
      ↓
Model Training & Evaluation (Accuracy, Precision, Recall, F1)
      ↓
Visualization (Distribution, Accuracy Comparison, WordCloud, Confusion Matrix)
```

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/alvin4104/student-hate-speech-detection.git
cd student-hate-speech-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

## Output Files

| File | Description |
|------|-------------|
| `results.png` | Label distribution, model accuracy, WordCloud |
| `confusion_matrix.png` | Confusion matrix of best model |

## Install Dependencies

```bash
pip install pandas numpy scikit-learn wordcloud matplotlib seaborn
```

## Dataset

Using built-in sample data for demonstration.  
For full experiments, download: [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
