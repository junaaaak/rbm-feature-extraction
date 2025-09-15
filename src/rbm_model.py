"""
RBM implementation and feature extraction functions.
"""

import time
from sklearn.neural_network import BernoulliRBM

def train_rbm(X_train_bin, n_components=64, learning_rate=0.05, 
              batch_size=10, n_iter=20, random_state=42):
    """
    Train a Restricted Boltzmann Machine.
    
    Args:
        X_train_bin: Binarized training data
        n_components: Number of hidden units
        learning_rate: Learning rate for RBM
        batch_size: Batch size for training
        n_iter: Number of iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Trained RBM model
    """
    # Create RBM
    rbm = BernoulliRBM(
        n_components=n_components,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_iter=n_iter,
        verbose=1,
        random_state=random_state
    )
    
    # Train RBM
    print("Training RBM...")
    start_time = time.time()
    rbm.fit(X_train_bin)
    training_time = time.time() - start_time
    print(f"RBM training completed in {training_time:.2f} seconds")
    
    return rbm

def extract_features(rbm, X_train_bin, X_test_bin):
    """
    Extract features using a trained RBM.
    
    Args:
        rbm: Trained RBM model
        X_train_bin: Binarized training data
        X_test_bin: Binarized test data
        
    Returns:
        Extracted features for training and test sets
    """
    # Extract features
    X_train_rbm = rbm.transform(X_train_bin)
    X_test_rbm = rbm.transform(X_test_bin)
    
    print(f"Original feature shape: {X_train_bin.shape}")
    print(f"RBM feature shape: {X_train_rbm.shape}")
    
    return X_train_rbm, X_test_rbm