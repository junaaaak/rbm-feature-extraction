"""
RBM Feature Extraction Pipeline - Main Script

This script orchestrates the complete pipeline for:
1. Loading and preprocessing the Digits dataset
2. Training a Restricted Boltzmann Machine (RBM)
3. Extracting features using the trained RBM
4. Comparing classification models with original vs RBM features
5. Comparing clustering algorithms with original vs RBM features
6. Generating visualizations and saving results

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import time
import argparse
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
from data_loading import load_and_preprocess_data
from rbm_model import train_rbm, extract_features
from classification import run_classification_experiments
from clustering import run_clustering_experiments
from visualization import plot_results, save_results

def setup_directories():
    """Create necessary directories for results if they don't exist."""
    directories = [
        'results/figures',
        'results/metrics',
        'data/digits_dataset'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RBM Feature Extraction Pipeline')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--n_components', type=int, default=64,
                        help='Number of hidden units in RBM (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Learning rate for RBM (default: 0.05)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for RBM training (default: 10)')
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Number of iterations for RBM training (default: 20)')
    parser.add_argument('--skip_clustering', action='store_true',
                        help='Skip clustering experiments to save time')
    parser.add_argument('--skip_classification', action='store_true',
                        help='Skip classification experiments to save time')
    
    return parser.parse_args()

def main():
    """Main function to run the complete RBM feature extraction pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Start timer
    start_time = time.time()
    
    # Print welcome message
    print("=" * 60)
    print("RBM FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    print("This pipeline will:")
    print("1. Load and preprocess the Digits dataset")
    print("2. Train a Restricted Boltzmann Machine (RBM)")
    print("3. Extract features using the trained RBM")
    print("4. Compare classification models with original vs RBM features")
    print("5. Compare clustering algorithms with original vs RBM features")
    print("6. Generate visualizations and save results")
    print("=" * 60)
    
    # Setup directories
    print("\nSetting up directories...")
    setup_directories()
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    data = load_and_preprocess_data(
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Train RBM and extract features
    print("\nTraining RBM...")
    rbm = train_rbm(
        data['X_train_bin'],
        n_components=args.n_components,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        random_state=args.random_state
    )
    
    # Extract features using the trained RBM
    print("\nExtracting features using RBM...")
    X_train_rbm, X_test_rbm = extract_features(
        rbm, 
        data['X_train_bin'], 
        data['X_test_bin']
    )
    
    # Add RBM features to data dictionary
    data['X_train_rbm'] = X_train_rbm
    data['X_test_rbm'] = X_test_rbm
    
    # Run classification experiments
    classification_results = {}
    if not args.skip_classification:
        print("\nRunning classification experiments...")
        classification_results = run_classification_experiments(data)
    else:
        print("\nSkipping classification experiments...")
    
    # Run clustering experiments
    clustering_results = {}
    if not args.skip_clustering:
        print("\nRunning clustering experiments...")
        clustering_results = run_clustering_experiments(data)
    else:
        print("\nSkipping clustering experiments...")
    
    # Generate and save visualizations
    print("\nGenerating visualizations...")
    if classification_results or clustering_results:
        summary_df = plot_results(classification_results, clustering_results)
        save_results(classification_results, clustering_results)
        
        # Display summary of results
        print("\n" + "=" * 60)
        print("SUMMARY OF RESULTS")
        print("=" * 60)
        print(summary_df.to_string(index=False))
    else:
        print("No results to visualize (both classification and clustering were skipped)")
    
    # Calculate and display total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Print completion message
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Results have been saved to the following directories:")
    print("- Visualizations: results/figures/")
    print("- Metrics: results/metrics/")
    print("=" * 60)

if __name__ == "__main__":
    main()