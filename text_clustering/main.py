"""
main.py
-------
This is the main script.
It performs the following steps:
    1. Data Loading: Loads documents from the specified directories.
    2. Preprocessing and TF-IDF Vectorization: Processes text and vectorizes documents.
    3. Clustering: Uses KMeans to cluster the TF-IDF feature matrix.
    4. Mapping and Evaluation: Maps predicted clusters to true labels and evaluates performance.
    5. Error Analysis: Identifies false positive and false negative examples for each cluster.
    6. Visualize Clusters
"""

from visualization import visualize_clusters
from data_loader import load_documents
from clustering import (
    perform_tfidf_vectorization, 
    perform_clustering, 
    map_clusters, 
    evaluate_clustering, 
    error_analysis
)

def main():

    print("Step 1: Loading Documents")
    # Define your directory paths
    dir_paths = [
        "3d_printing",
        "filter_bubble",
        "indoor_positioning_in_cultural_heritage",
        "twitter_bias"
    ]
    
    # Load documents into a DataFrame
    df = load_documents(dir_paths)
    print(f"Loaded {len(df)} documents.")
    
    print("\nStep 2: Preprocessing and TF-IDF Vectorization")
    # Preprocess and vectorize documents
    X, vectorizer = perform_tfidf_vectorization(df)
    print("TF-IDF vectorization complete.")
    
    print("\nStep 3.1: Clustering with KMeans")
    # Cluster the TF-IDF matrix using KMeans
    predicted_clusters = perform_clustering(X)
    df['predicted'] = predicted_clusters
    print("Clustering complete.")
    
    print("\nStep 3.2: Mapping Clusters to True Labels")
    # Map predicted clusters to true labels
    true_labels = df['label'].values
    mapping, mapped_preds, cont_matrix = map_clusters(true_labels, predicted_clusters)
    print("Mapping complete.")
    
    print("\nStep 3.3: Evaluating Clustering")
    # Evaluate the clustering
    metrics = evaluate_clustering(true_labels, predicted_clusters, mapped_preds, cont_matrix)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print("\nStep 3.4: Error Analysis")
    # Perform error analysis for each cluster
    errors = error_analysis(df, mapped_preds, mapping)
    for cluster, error in errors.items():
        print(f"\nFor Cluster (mapped as '{cluster}'):")
        if error["false_positive"] is not None:
            print("  False Positive example:")
            print("    Document snippet:", error["false_positive"]['document'][:100])
        else:
            print("  No false positives found.")
        if error["false_negative"] is not None:
            print("  False Negative example:")
            print("    Document snippet:", error["false_negative"]['document'][:100])
        else:
            print("  No false negatives found.")

    # (Optionally) Step 4: Visualize Clusters
    #print("\nStep 4: Visualizing Clusters using t-SNE")
    #visualize_clusters(X, mapped_preds)
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()