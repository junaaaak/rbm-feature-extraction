"""
Clustering algorithms and evaluation functions.
"""

import time
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

def run_clustering_experiments(data):
    """
    Run clustering experiments with original and RBM features.
    
    Args:
        data: Dictionary containing preprocessed data
        
    Returns:
        Dictionary with clustering results
    """
    results = {}
    
    # K-Means on original features
    print("Running K-Means on original features...")
    kmeans_original = KMeans(n_clusters=10, random_state=42, n_init=10)
    start_time = time.time()
    kmeans_original_labels = kmeans_original.fit_predict(data['X_train_std'])
    kmeans_original_time = time.time() - start_time
    kmeans_original_ari = adjusted_rand_score(data['y_train'], kmeans_original_labels)
    kmeans_original_sil = silhouette_score(data['X_train_std'], kmeans_original_labels)
    
    # K-Means on RBM features
    print("Running K-Means on RBM features...")
    kmeans_rbm = KMeans(n_clusters=10, random_state=42, n_init=10)
    start_time = time.time()
    kmeans_rbm_labels = kmeans_rbm.fit_predict(data['X_train_rbm'])
    kmeans_rbm_time = time.time() - start_time
    kmeans_rbm_ari = adjusted_rand_score(data['y_train'], kmeans_rbm_labels)
    kmeans_rbm_sil = silhouette_score(data['X_train_rbm'], kmeans_rbm_labels)
    
    # DBSCAN on original features
    print("Running DBSCAN on original features...")
    dbscan_original = DBSCAN(eps=3.5, min_samples=5)
    start_time = time.time()
    dbscan_original_labels = dbscan_original.fit_predict(data['X_train_std'])
    dbscan_original_time = time.time() - start_time
    # Only calculate metrics if we have more than one cluster
    if len(np.unique(dbscan_original_labels)) > 1:
        dbscan_original_ari = adjusted_rand_score(data['y_train'], dbscan_original_labels)
        dbscan_original_sil = silhouette_score(data['X_train_std'], dbscan_original_labels)
    else:
        dbscan_original_ari = -1
        dbscan_original_sil = -1
    
    # DBSCAN on RBM features
    print("Running DBSCAN on RBM features...")
    dbscan_rbm = DBSCAN(eps=2.5, min_samples=5)
    start_time = time.time()
    dbscan_rbm_labels = dbscan_rbm.fit_predict(data['X_train_rbm'])
    dbscan_rbm_time = time.time() - start_time
    # Only calculate metrics if we have more than one cluster
    if len(np.unique(dbscan_rbm_labels)) > 1:
        dbscan_rbm_ari = adjusted_rand_score(data['y_train'], dbscan_rbm_labels)
        dbscan_rbm_sil = silhouette_score(data['X_train_rbm'], dbscan_rbm_labels)
    else:
        dbscan_rbm_ari = -1
        dbscan_rbm_sil = -1
    
    # Store results
    results['kmeans'] = {
        'original': {
            'ari': kmeans_original_ari, 
            'silhouette': kmeans_original_sil, 
            'time': kmeans_original_time
        },
        'rbm': {
            'ari': kmeans_rbm_ari, 
            'silhouette': kmeans_rbm_sil, 
            'time': kmeans_rbm_time
        }
    }
    
    results['dbscan'] = {
        'original': {
            'ari': dbscan_original_ari, 
            'silhouette': dbscan_original_sil, 
            'time': dbscan_original_time
        },
        'rbm': {
            'ari': dbscan_rbm_ari, 
            'silhouette': dbscan_rbm_sil, 
            'time': dbscan_rbm_time
        }
    }
    
    # Print results
    print("\nClustering Results:")
    print("=" * 50)
    print(f"K-Means - Original: ARI = {kmeans_original_ari:.4f}, Silhouette = {kmeans_original_sil:.4f}, Time = {kmeans_original_time:.4f}s")
    print(f"K-Means - RBM: ARI = {kmeans_rbm_ari:.4f}, Silhouette = {kmeans_rbm_sil:.4f}, Time = {kmeans_rbm_time:.4f}s")
    print(f"DBSCAN - Original: ARI = {dbscan_original_ari:.4f}, Silhouette = {dbscan_original_sil:.4f}, Time = {dbscan_original_time:.4f}s")
    print(f"DBSCAN - RBM: ARI = {dbscan_rbm_ari:.4f}, Silhouette = {dbscan_rbm_sil:.4f}, Time = {dbscan_rbm_time:.4f}s")
    
    return results