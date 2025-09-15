# RBM Feature Extraction for Digits Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a Restricted Boltzmann Machine (RBM) for feature extraction on the Digits dataset from scikit-learn. It compares the performance of classification and clustering algorithms using both original features and RBM-extracted features.

## 📊 Project Overview

The project explores how feature extraction using Restricted Boltzmann Machines can impact the performance of machine learning tasks. Specifically, it compares:

**Classification Models:**
- Logistic Regression with original vs RBM features
- Neural Network with original vs RBM features

**Clustering Algorithms:**
- K-Means with original vs RBM features
- DBSCAN with original vs RBM features

## 🚀 Installation

Clone the repository:
```bash
git clone https://github.com/junaaaak/rbm-feature-extraction.git
cd rbm-feature-extraction
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
rbm-feature-extraction/
├── data/                     # Dataset directory (auto-created)
├── notebooks/                # Jupyter notebooks for analysis
│   └── rbm_digits_analysis.ipynb
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_loading.py       # Data loading and preprocessing
│   ├── rbm_model.py         # RBM implementation and training
│   ├── classification.py    # Classification models
│   ├── clustering.py        # Clustering algorithms
│   └── visualization.py     # Visualization utilities
├── results/                  # Output directory (auto-created)
│   ├── figures/             # Saved plots and visualizations
│   └── metrics/             # Saved performance metrics
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── main.py                 # Main execution script
```

## 🎯 Usage

Run the complete analysis pipeline:
```bash
python main.py
```

Explore the analysis notebook:
```bash
jupyter notebook notebooks/rbm_digits_analysis.ipynb
```

Import and use specific modules:
```python
from src.data_loading import load_and_preprocess_data
from src.rbm_model import train_rbm, extract_features
from src.classification import run_classification_experiments
from src.clustering import run_clustering_experiments

# Load data
data = load_and_preprocess_data()

# Train RBM
rbm = train_rbm(data['X_train_bin'])

# Extract features
X_train_rbm, X_test_rbm = extract_features(rbm, data['X_train_bin'], data['X_test_bin'])
```

## 🔬 Methodology

1. **Data Preparation**: The Digits dataset is loaded and preprocessed by binarizing pixel values for RBM training and standardizing for traditional algorithms.

2. **RBM Training**: A Restricted Boltzmann Machine with 64 hidden units is trained on the binarized data to learn a new feature representation.

3. **Feature Extraction**: The trained RBM is used to transform both training and test data into the new feature space.

4. **Model Comparison**: Classification and clustering algorithms are applied to both the original and RBM-transformed features, with performance metrics recorded for comparison.

5. **Visualization**: Results are visualized through comparative plots and saved for analysis.

## 📈 Results

The project generates comprehensive comparisons of:

- Classification accuracy for Logistic Regression and Neural Networks
- Clustering performance metrics (Adjusted Rand Index and Silhouette Score) for K-Means and DBSCAN
- Training times for all algorithms with both feature types

Sample findings typically show:
- RBM feature extraction can improve performance for certain algorithms
- The effectiveness depends on the specific model and task
- RBM features often create more separable representations for clustering tasks

## 📋 Dependencies

- Python 3.8+
- numpy
- scikit-learn
- matplotlib
- pandas
- jupyter

See `requirements.txt` for complete list with versions.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
