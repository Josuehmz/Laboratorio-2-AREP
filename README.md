# Heart Disease Risk Prediction: Logistic Regression

## Overview
Implementation of logistic regression from scratch to predict heart disease risk using clinical features.

## Dataset
- **Source**: Heart Disease Prediction Dataset (Kaggle)
- **Samples**: 270 records
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

## Current Implementation

### Step 1: Load and Prepare the Dataset
- Dataset loading and exploration
- Target variable binarization (Presence → 1, Absence → 0)
- Exploratory Data Analysis (EDA)
- Feature selection (6 features)
- Class distribution visualization
- Correlation analysis

### Step 2: Implement Basic Logistic Regression
- **Sigmoid function**: Activation function implementation
- **Cost function**: Binary cross-entropy (log loss) computation
- **Gradient computation**: Calculation of weight and bias gradients
- **Gradient Descent**: Optimization algorithm with cost tracking

#### Training
- **Data split**: Stratified 70/30 train/test split
- **Feature normalization**: StandardScaler for numerical features
- **Parameter initialization**: Weights and bias initialized to zero
- **Model training**: 
  - Learning rate (α) = 0.01
  - 2000 iterations (1000+)
  - Training on full training set
  - Cost tracking during optimization
- **Cost visualization**: Plot of cost vs iterations showing convergence



