"""
clustering.py
-------------
This module handles the TF-IDF vectorization, clustering, mapping of predicted clusters to true labels,
evaluation of clustering performance, and error analysis.

Functions:
    perform_tfidf_vectorization(documents_df):
        Vectorizes documents using TF-IDF and returns the feature matrix and vectorizer.
    
    perform_clustering(X, n_clusters=4, random_state=42):
        Performs KMeans clustering on the TF-IDF matrix and returns predicted cluster labels.
    
    map_clusters(true_labels, predicted_clusters):
        Maps the predicted clusters to the true labels using the Hungarian algorithm.
    
    evaluate_clustering(true_labels, predicted_clusters, mapped_preds, cont_matrix):
        Computes and returns evaluation metrics (accuracy, precision, recall, F1, purity, ARI).
    
    error_analysis(documents_df, mapped_preds, mapping):
        Performs error analysis by identifying a false positive and a false negative for each cluster.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from preprocess import custom_preprocessor

def perform_tfidf_vectorization(documents_df):
    """
    Vectorizes the documents using TF-IDF.

    Parameters:
        documents_df (DataFrame): DataFrame containing the documents in a 'document' column.
    
    Returns:
        X (sparse matrix): TF-IDF feature matrix.
        vectorizer (TfidfVectorizer): The fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=custom_preprocessor,
        token_pattern=r'\b\w+\b'
    )
    X = vectorizer.fit_transform(documents_df['document'])
    return X, vectorizer

def perform_clustering(X, n_clusters=4, random_state=42):
    """
    Clusters the TF-IDF feature matrix using KMeans.

    Parameters:
        X (sparse matrix): TF-IDF feature matrix.
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        predicted_clusters (ndarray): Array of cluster labels for each document.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    predicted_clusters = kmeans.fit_predict(X)
    return predicted_clusters

def map_clusters(true_labels, predicted_clusters):
    """
    Maps predicted cluster labels to true labels using the Hungarian algorithm.

    Parameters:
        true_labels (array-like): Array of true labels.
        predicted_clusters (array-like): Array of predicted cluster labels.
    
    Returns:
        mapping (dict): Dictionary mapping predicted cluster number to true label.
        mapped_preds (ndarray): Array of predicted labels after mapping.
        cont_matrix (ndarray): Contingency matrix used for mapping.
    """
    cont_matrix = contingency_matrix(true_labels, predicted_clusters)
    row_ind, col_ind = linear_sum_assignment(-cont_matrix)
    unique_true_labels = np.unique(true_labels)
    mapping = {col: unique_true_labels[row] for row, col in zip(row_ind, col_ind)}
    mapped_preds = np.array([mapping.get(cluster, cluster) for cluster in predicted_clusters])
    return mapping, mapped_preds, cont_matrix

def evaluate_clustering(true_labels, predicted_clusters, mapped_preds, cont_matrix):
    """
    Computes evaluation metrics for the clustering.

    Parameters:
        true_labels (array-like): True labels.
        predicted_clusters (array-like): Raw predicted cluster labels.
        mapped_preds (array-like): Predicted labels after mapping.
        cont_matrix (ndarray): Contingency matrix.
    
    Returns:
        metrics (dict): Dictionary of evaluation metrics (accuracy, precision, recall, F1, purity, ARI).
    """
    accuracy = accuracy_score(true_labels, mapped_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, mapped_preds, average='weighted')
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    purity = np.sum(np.amax(cont_matrix, axis=0)) / np.sum(cont_matrix)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "purity": purity,
        "ari": ari
    }
    return metrics

def error_analysis(documents_df, mapped_preds, mapping):
    """
    Performs error analysis by identifying a false positive and a false negative example for each cluster.

    Parameters:
        documents_df (DataFrame): DataFrame containing documents and their true labels.
        mapped_preds (ndarray): Predicted labels after mapping.
        mapping (dict): Mapping from raw predicted clusters to true labels.
    
    Returns:
        error_info (dict): Dictionary containing error examples for each cluster.
                           Keys are the mapped cluster labels and values are dictionaries with keys:
                           'false_positive' and 'false_negative'.
    """
    error_info = {}
    for cluster in np.unique(mapped_preds):
        pred_idx = documents_df.index[mapped_preds == cluster].tolist()
        true_idx = documents_df.index[documents_df['label'] == cluster].tolist()

        false_positives = [i for i in pred_idx if i not in true_idx]
        false_negatives = [i for i in true_idx if i not in pred_idx]

        error_info[cluster] = {
            "false_positive": documents_df.loc[false_positives[0]] if false_positives else None,
            "false_negative": documents_df.loc[false_negatives[0]] if false_negatives else None
        }
    return error_info
