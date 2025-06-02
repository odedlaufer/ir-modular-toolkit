"""
result_analysis.py

This module provides functions for:
- Identifying misclassified examples.
- Extracting top words influencing classification.
- Printing misclassified document details.

Functions:
    - get_misclassified_examples(): Finds and prints misclassified examples.
    - analyze_feature_importance(): Extracts top words for Naive Bayes & Logistic Regression.

Usage:
    Import these functions after training models in `model_training.py`.

Example:
    from result_analysis import get_misclassified_examples, analyze_feature_importance
    get_misclassified_examples(df, all_preds, all_true)
    analyze_feature_importance(vectorizer, models)
"""

import numpy as np

def get_misclassified_examples(df, all_preds, all_true, num_examples=10):
    """
    Identifies and prints misclassified examples from the dataset.

    Args:
        df (pd.DataFrame): The dataset containing filenames and processed text.
        all_preds (list): List of model predictions.
        all_true (list): List of actual labels.
        num_examples (int): Number of misclassified examples to display (default: 5).
    """
    misclassified_indices = np.where(np.array(all_preds) != np.array(all_true))[0]

    print("\nMisclassified Examples:")
    for idx in misclassified_indices[:num_examples]:
        if idx >= len(df):  
            continue

        print(f"Filename: {df.iloc[idx]['filename']}")
        print(f"Actual Label: {df.iloc[idx]['label']} | Predicted: {all_preds[idx]}")
        print(f"Text Snippet: {df.iloc[idx]['processed_text'][:300]}")  # First 300 chars
        print("-" * 50)


def analyze_feature_importance(vectorizer, models):
    """
    Extracts and prints top words influencing classification for Naive Bayes and Logistic Regression.

    Args:
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer used to transform the dataset.
        models (dict): Dictionary containing trained models.

    Outputs:
        - Prints top words associated with each class.
    """
    feature_names = vectorizer.get_feature_names_out()

    # Naive Bayes feature analysis
    if "Naive Bayes" in models:
        class_probabilities = models["Naive Bayes"].feature_log_prob_
        top_words_positive = np.argsort(class_probabilities[1])[-10:]
        top_words_negative = np.argsort(class_probabilities[0])[-10:]

        print("\nNaive Bayes - Top Words per Class")
        print("Top words predicting category (1):")
        print([feature_names[i] for i in top_words_positive])

        print("\nTop words predicting non-category (0):")
        print([feature_names[i] for i in top_words_negative])

    # Logistic Regression feature analysis
    if "Logistic Regression" in models:
        coefs = models["Logistic Regression"].coef_[0]
        top_positive = np.argsort(coefs)[-10:]
        top_negative = np.argsort(coefs)[:10]

        print("\nLogistic Regression - Top Words per Class")
        print("Top positive words (category 1):")
        print([feature_names[i] for i in top_positive])

        print("\nTop negative words (category 0):")
        print([feature_names[i] for i in top_negative])