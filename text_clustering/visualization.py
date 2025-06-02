"""
visualization.py
----------------

Functions:
    visualize_clusters(X, mapped_preds):
        Reduces the dimensionality of X and plots the clusters with annotated cluster names.
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import numpy as np

def visualize_clusters(X, mapped_preds):
    """
    Visualize clusters using t-SNE.
    
    Parameters:
        X (sparse matrix): The TF-IDF feature matrix.
        mapped_preds (array-like): The predicted cluster labels (as strings) after mapping.
    """
    # Convert mapped_preds (strings) to numeric values for visualization
    encoder = LabelEncoder()
    mapped_preds_numeric = encoder.fit_transform(mapped_preds)
    
    X_dense = X.toarray()  
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_reduced = tsne.fit_transform(X_dense)
    
    # Create a scatter plot using numeric cluster labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                          c=mapped_preds_numeric, cmap='viridis', alpha=0.7)
    
    plt.title("t-SNE Visualization of Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    cbar = plt.colorbar(scatter, label="Encoded Cluster Label")
    
    unique_numeric_labels = np.unique(mapped_preds_numeric)
    for label_num in unique_numeric_labels:
        indices = np.where(mapped_preds_numeric == label_num)[0]
        # Calculate the center points
        center_x = np.mean(X_reduced[indices, 0])
        center_y = np.mean(X_reduced[indices, 1])
        # Get the original label corresponding to label
        cluster_name = encoder.inverse_transform([label_num])[0]
        plt.text(center_x, center_y, cluster_name, fontsize=12, fontweight='bold',
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    
    plt.show()
