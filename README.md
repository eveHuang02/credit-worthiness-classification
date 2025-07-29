# Classifying Credit Worthiness 

## Project Description
Classifying credit worthiness using big data mining techniques on over 850,000 loan records. 
Applied PySpark and TensorFlow to build Decision Tree, Random Forest, Gradient Boosted, and Neural Network models.
Achieved up to **99.93% accuracy** and **0.9999 AUC**, significantly improving the identification of high-risk applicants.

## Features
- Handled missing values, type inconsistencies, imbalanced categories
- Dropped non-informative features and selected relevant features via correlation and feature importance
- Performed one-hot encoding and feature scaling
- Built multiple machine learning models using PySpark and TensorFlow
- Tuned hyperparameters using grid search and cross-validation
- Evaluated models using accuracy, precision, recall, F1-score, and AUC

## Dataset
- Original dataset contains 855,969 loan applications and 73 features
- After preprocessing: 846,317 records, no missing values, and reduced feature dimensionality

## Model Summary

| Model               | Framework     | Accuracy | AUC     |
|--------------------|---------------|----------|---------|
| Random Forest       | PySpark       | 99.93%   | 0.9999  |
| Gradient Boosted    | TensorFlow    | 99.79%   | 0.9977  |
| Neural Network      | TensorFlow    | 99.78%   | 0.998   |
| Decision Tree       | PySpark       | ~99.19%  | ~0.991  |

## Folder Structure
```
credit-worthiness-classification/
├── code.ipynb        # Main notebook (PySpark and TensorFlow modeling)
├── data/
│   └── data.zip             # Raw dataset 
├── README.md                
```

## Getting Started

### Prerequisites
- Python 3.8+
- Apache Spark
- PySpark
- TensorFlow
- pandas, numpy, matplotlib, scikit-learn

### Installation
1. Clone the repo
```
git clone https://github.com/your-username/credit-worthiness-classification.git
cd credit-worthiness-classification
```

2. Run Jupyter Notebook
```
jupyter notebook code.ipynb
```

## Results & Visuals
- Confusion matrices, correlation heatmaps, feature importance plots, and accuracy comparisons were used.
- TensorFlow and PySpark both achieved excellent performance, especially after hyperparameter tuning.