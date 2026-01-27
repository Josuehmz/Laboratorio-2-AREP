# Heart Disease Risk Prediction: Logistic Regression

## Overview
Implementation of logistic regression from scratch to predict heart disease risk using clinical features.

## Dataset
- **Source**: Heart Disease Prediction Dataset (Kaggle)
- **Features**: 6 selected features (Age, Cholesterol, BP, Max HR, ST depression, Number of vessels fluro)
- **Target**: Binary classification (Presence/Absence of heart disease)

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## Usage
1. Open `Heart_Disease_Prediction.ipynb` in Jupyter Notebook
2. Run all cells sequentially

## Implementation
- Sigmoid function
- Binary cross-entropy cost function
- Gradient descent optimization
- Training with α = 0.01, 2000 iterations
- 70/30 train/test split (stratified)

## Results
- Model performance metrics (Accuracy, Precision, Recall, F1-Score)
- Cost vs iterations visualization
- Confusion matrices
- Feature importance analysis
