"""
model_training.py

This module:
- Loads preprocessed text data.
- Converts text into TF-IDF vectors.
- Trains multiple classifiers using 10-fold cross-validation.
- Evaluates models using accuracy, precision, recall, and F1-score.

Functions:
    - train_and_evaluate_models(): Runs model training and evaluation.

Usage:
    import `train_and_evaluate_models()` in `main.py`.

Example:
    from model_training import train_and_evaluate_models
    train_and_evaluate_models()
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from config import PROCESSED_CSV
from result_analysis import get_misclassified_examples, analyze_feature_importance

def load_data():
    """
    Loads preprocessed dataset and vectorizes text using TF-IDF.

    Returns:
        X_vectorized (sparse matrix): TF-IDF feature matrix.
        y (pd.Series): Labels (0 or 1).
        df (pd.DataFrame): Dataframe containing processed text and labels.
        vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
    """
    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_CSV)
    X = df["processed_text"]
    y = df["label"].astype(int)

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    return X_vectorized, y, df, vectorizer


def train_and_evaluate_models():
    """
    Trains multiple classifiers using 10-fold cross-validation and evaluates them.
    
    Returns:
        df (pd.DataFrame): Dataframe containing processed text and labels.
        all_preds (list): List of model predictions from the last model trained.
        all_true (list): List of actual labels.
        vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        trained_models (dict): Dictionary containing trained models.
    """
    X_vectorized, y, df, vectorizer = load_data()

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Linear)": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    trained_models = {}  
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        all_preds = []  # Reset predictions
        all_true = []   # Reset ground truth labels

        for train_idx, test_idx in cv.split(X_vectorized, y):
            X_train, X_test = X_vectorized[train_idx], X_vectorized[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            all_preds.extend(preds)
            all_true.extend(y_test)

        trained_models[model_name] = model  

        cm = confusion_matrix(all_true, all_preds)
        acc = accuracy_score(all_true, all_preds)
        precision = precision_score(all_true, all_preds, zero_division=0)
        recall = recall_score(all_true, all_preds, zero_division=0)
        f1 = f1_score(all_true, all_preds, zero_division=0)

        print(f"\n=== {model_name} ===")
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")

    return df, all_preds, all_true, vectorizer, trained_models

