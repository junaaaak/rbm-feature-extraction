"""
Classification models and evaluation functions.
"""

import time
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def run_classification_experiments(data):
    """
    Run classification experiments with original and RBM features.
    
    Args:
        data: Dictionary containing preprocessed data
        
    Returns:
        Dictionary with classification results
    """
    results = {}
    
    # Logistic Regression on original features
    print("Training Logistic Regression on original features...")
    lr_original = LogisticRegression(max_iter=1000, random_state=42)
    start_time = time.time()
    lr_original.fit(data['X_train_std'], data['y_train'])
    lr_original_time = time.time() - start_time
    lr_original_pred = lr_original.predict(data['X_test_std'])
    lr_original_acc = accuracy_score(data['y_test'], lr_original_pred)
    
    # Logistic Regression on RBM features
    print("Training Logistic Regression on RBM features...")
    lr_rbm = LogisticRegression(max_iter=1000, random_state=42)
    start_time = time.time()
    lr_rbm.fit(data['X_train_rbm'], data['y_train'])
    lr_rbm_time = time.time() - start_time
    lr_rbm_pred = lr_rbm.predict(data['X_test_rbm'])
    lr_rbm_acc = accuracy_score(data['y_test'], lr_rbm_pred)
    
    # Neural Network on original features
    print("Training Neural Network on original features...")
    nn_original = MLPClassifier(
        hidden_layer_sizes=(100,), 
        activation='relu', 
        max_iter=500, 
        random_state=42
    )
    start_time = time.time()
    nn_original.fit(data['X_train_std'], data['y_train'])
    nn_original_time = time.time() - start_time
    nn_original_pred = nn_original.predict(data['X_test_std'])
    nn_original_acc = accuracy_score(data['y_test'], nn_original_pred)
    
    # Neural Network on RBM features
    print("Training Neural Network on RBM features...")
    nn_rbm = MLPClassifier(
        hidden_layer_sizes=(100,), 
        activation='relu', 
        max_iter=500, 
        random_state=42
    )
    start_time = time.time()
    nn_rbm.fit(data['X_train_rbm'], data['y_train'])
    nn_rbm_time = time.time() - start_time
    nn_rbm_pred = nn_rbm.predict(data['X_test_rbm'])
    nn_rbm_acc = accuracy_score(data['y_test'], nn_rbm_pred)
    
    # Store results
    results['logistic_regression'] = {
        'original': {'accuracy': lr_original_acc, 'time': lr_original_time},
        'rbm': {'accuracy': lr_rbm_acc, 'time': lr_rbm_time}
    }
    
    results['neural_network'] = {
        'original': {'accuracy': nn_original_acc, 'time': nn_original_time},
        'rbm': {'accuracy': nn_rbm_acc, 'time': nn_rbm_time}
    }
    
    # Print results
    print("\nClassification Results:")
    print("=" * 50)
    print(f"Logistic Regression - Original: Accuracy = {lr_original_acc:.4f}, Time = {lr_original_time:.4f}s")
    print(f"Logistic Regression - RBM: Accuracy = {lr_rbm_acc:.4f}, Time = {lr_rbm_time:.4f}s")
    print(f"Neural Network - Original: Accuracy = {nn_original_acc:.4f}, Time = {nn_original_time:.4f}s")
    print(f"Neural Network - RBM: Accuracy = {nn_rbm_acc:.4f}, Time = {nn_rbm_time:.4f}s")
    
    return results