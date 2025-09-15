"""
Data loading and preprocessing functions.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Load and preprocess the digits dataset.
    
    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing preprocessed data
    """
    # Load the digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Preprocess the data
    # Binarize for RBM (convert to binary values)
    binarizer = Binarizer(threshold=0.5)
    X_train_bin = binarizer.fit_transform(X_train / 16.0)  # Scale to [0,1] then binarize
    X_test_bin = binarizer.transform(X_test / 16.0)
    
    # Standardize for traditional methods
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Return all data in a dictionary
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_bin': X_train_bin,
        'X_test_bin': X_test_bin,
        'X_train_std': X_train_std,
        'X_test_std': X_test_std
    }