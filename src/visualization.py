"""
Visualization functions for results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_results(classification_results, clustering_results):
    """
    Create visualizations of the results.
    
    Args:
        classification_results: Dictionary with classification results
        clustering_results: Dictionary with clustering results
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Classification accuracy comparison
    models = ['Logistic Regression', 'Neural Network']
    original_acc = [
        classification_results['logistic_regression']['original']['accuracy'],
        classification_results['neural_network']['original']['accuracy']
    ]
    rbm_acc = [
        classification_results['logistic_regression']['rbm']['accuracy'],
        classification_results['neural_network']['rbm']['accuracy']
    ]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, original_acc, width, label='Original Features')
    axes[0, 0].bar(x + width/2, rbm_acc, width, label='RBM Features')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Classification Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Clustering ARI comparison
    algorithms = ['K-Means', 'DBSCAN']
    original_ari = [
        clustering_results['kmeans']['original']['ari'],
        max(0, clustering_results['dbscan']['original']['ari'])  # Handle negative values
    ]
    rbm_ari = [
        clustering_results['kmeans']['rbm']['ari'],
        max(0, clustering_results['dbscan']['rbm']['ari'])  # Handle negative values
    ]
    
    axes[0, 1].bar(x - width/2, original_ari, width, label='Original Features')
    axes[0, 1].bar(x + width/2, rbm_ari, width, label='RBM Features')
    axes[0, 1].set_xlabel('Algorithms')
    axes[0, 1].set_ylabel('Adjusted Rand Index')
    axes[0, 1].set_title('Clustering ARI Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(algorithms)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Clustering Silhouette comparison
    original_sil = [
        clustering_results['kmeans']['original']['silhouette'],
        max(0, clustering_results['dbscan']['original']['silhouette'])  # Handle negative values
    ]
    rbm_sil = [
        clustering_results['kmeans']['rbm']['silhouette'],
        max(0, clustering_results['dbscan']['rbm']['silhouette'])  # Handle negative values
    ]
    
    axes[1, 0].bar(x - width/2, original_sil, width, label='Original Features')
    axes[1, 0].bar(x + width/2, rbm_sil, width, label='RBM Features')
    axes[1, 0].set_xlabel('Algorithms')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].set_title('Clustering Silhouette Score Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(algorithms)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Training time comparison for classification
    original_time = [
        classification_results['logistic_regression']['original']['time'],
        classification_results['neural_network']['original']['time']
    ]
    rbm_time = [
        classification_results['logistic_regression']['rbm']['time'],
        classification_results['neural_network']['rbm']['time']
    ]
    
    axes[1, 1].bar(x - width/2, original_time, width, label='Original Features')
    axes[1, 1].bar(x + width/2, rbm_time, width, label='RBM Features')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Classification Training Time Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/figures/comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary table
    summary_data = {
        'Model': ['Logistic Regression (Original)', 'Logistic Regression (RBM)',
                 'Neural Network (Original)', 'Neural Network (RBM)',
                 'K-Means (Original)', 'K-Means (RBM)',
                 'DBSCAN (Original)', 'DBSCAN (RBM)'],
        'Accuracy/ARI': [
            classification_results['logistic_regression']['original']['accuracy'],
            classification_results['logistic_regression']['rbm']['accuracy'],
            classification_results['neural_network']['original']['accuracy'],
            classification_results['neural_network']['rbm']['accuracy'],
            clustering_results['kmeans']['original']['ari'],
            clustering_results['kmeans']['rbm']['ari'],
            clustering_results['dbscan']['original']['ari'],
            clustering_results['dbscan']['rbm']['ari']
        ],
        'Silhouette (Clustering Only)': [
            '-', '-', '-', '-',
            clustering_results['kmeans']['original']['silhouette'],
            clustering_results['kmeans']['rbm']['silhouette'],
            clustering_results['dbscan']['original']['silhouette'],
            clustering_results['dbscan']['rbm']['silhouette']
        ],
        'Time (seconds)': [
            classification_results['logistic_regression']['original']['time'],
            classification_results['logistic_regression']['rbm']['time'],
            classification_results['neural_network']['original']['time'],
            classification_results['neural_network']['rbm']['time'],
            clustering_results['kmeans']['original']['time'],
            clustering_results['kmeans']['rbm']['time'],
            clustering_results['dbscan']['original']['time'],
            clustering_results['dbscan']['rbm']['time']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def save_results(classification_results, clustering_results):
    """
    Save results to CSV files.
    
    Args:
        classification_results: Dictionary with classification results
        clustering_results: Dictionary with clustering results
    """
    # Save classification results
    cls_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Neural Network'],
        'Original Accuracy': [
            classification_results['logistic_regression']['original']['accuracy'],
            classification_results['neural_network']['original']['accuracy']
        ],
        'RBM Accuracy': [
            classification_results['logistic_regression']['rbm']['accuracy'],
            classification_results['neural_network']['rbm']['accuracy']
        ],
        'Original Time (s)': [
            classification_results['logistic_regression']['original']['time'],
            classification_results['neural_network']['original']['time']
        ],
        'RBM Time (s)': [
            classification_results['logistic_regression']['rbm']['time'],
            classification_results['neural_network']['rbm']['time']
        ]
    })
    cls_df.to_csv('results/metrics/classification_results.csv', index=False)
    
    # Save clustering results
    clst_df = pd.DataFrame({
        'Algorithm': ['K-Means', 'DBSCAN'],
        'Original ARI': [
            clustering_results['kmeans']['original']['ari'],
            clustering_results['dbscan']['original']['ari']
        ],
        'RBM ARI': [
            clustering_results['kmeans']['rbm']['ari'],
            clustering_results['dbscan']['rbm']['ari']
        ],
        'Original Silhouette': [
            clustering_results['kmeans']['original']['silhouette'],
            clustering_results['dbscan']['original']['silhouette']
        ],
        'RBM Silhouette': [
            clustering_results['kmeans']['rbm']['silhouette'],
            clustering_results['dbscan']['rbm']['silhouette']
        ],
        'Original Time (s)': [
            clustering_results['kmeans']['original']['time'],
            clustering_results['dbscan']['original']['time']
        ],
        'RBM Time (s)': [
            clustering_results['kmeans']['rbm']['time'],
            clustering_results['dbscan']['rbm']['time']
        ]
    })
    clst_df.to_csv('results/metrics/clustering_results.csv', index=False)
    
    print("Results saved to results/metrics/ directory")