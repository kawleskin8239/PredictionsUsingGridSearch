# PredictionsUsingGridSearch
## Overview
There are three parts to this project, each having their own objectives and using their own datasets

## 1. Gini Impurity Calculator
This section provides a utility function to calculate **Gini Impurity**, a metric commonly used in decision tree algorithms to evaluate the purity of a dataset split.

### Functionality
- Accepts a list of any size representing the count of samples in each class.
- Computes the Gini Impurity using the formula:

  ![image](https://github.com/user-attachments/assets/a4ba86cc-0ae1-4fd9-a084-bd206e3eb7ee)


  where \( c_i \) is the count of class \( i \), and \( N \) is the total number of samples.
- Prints the score to the console

## 2. Machine Maintenance Prediction
This project uses machine learning to predict machine maintenance needs from sensor and operational data. It applies a Decision Tree Classifier with hyperparameter tuning via GridSearchCV to optimize model performance.

### Dataset

The dataset used is **predictive_maintenance.csv**, which includes various sensor readings and categorical variables related to machine operation and failure status.

### Features

- Data loading and cleaning
- Conversion and encoding of categorical variables
- Train/test split
- Hyperparameter tuning for `max_depth` using GridSearchCV
- Classification using Decision Tree with balanced class weights
- Visualization of confusion matrix and decision tree structure

### Requirements

Make sure you have the following Python packages installed:

`pip install pandas scikit-learn matplotlib`
### Usage

1. Place `predictive_maintenance.csv` in the same directory.  
2. Run the Python script.

### Steps Performed

### Preprocessing
- Loads data and drops missing values.  
- Drops initial irrelevant columns.  
- Converts object columns to categorical and encodes them as numeric codes.  
- Prints a dictionary showing category encodings for categorical features.  
- Separates features and target labels.

### Train/Test Split
- Splits data into 70% training and 30% test sets.

### Hyperparameter Tuning
- Uses GridSearchCV to find the optimal `max_depth` for the Decision Tree.  
- Prints cross-validation mean test scores and ranks.

### Training and Evaluation
- Trains the Decision Tree with the best `max_depth` and balanced class weights.  
- Displays normalized confusion matrix.  
- Visualizes the decision tree structure limited to a depth of 2 for clarity.

### Results
- Dictionary of categorical variable encodings is printed to console.  
- Cross-validation scores for different tree depths are printed.  
- Confusion matrix plot for test set predictions.  
- Decision tree visualization plot (depth limited to 2).

## 3. CSGO Round Prediction
This project uses machine learning to predict the outcome of CSGO rounds based on game state snapshots. It applies Support Vector Classifier (SVC) and Random Forest models with hyperparameter tuning via GridSearchCV to optimize predictive performance.

### Dataset

The dataset used is **csgo_round_snapshots.csv**, containing game state features such as time left, scores, and player health, along with the round winner label.

### Features

- Data loading and cleaning
- Conversion of categorical labels to numeric
- Feature selection and normalization
- Train/test split
- Hyperparameter tuning for SVC (`C` and `gamma`) and Random Forest (`max_depth`) using GridSearchCV
- Classification using SVC with RBF kernel and balanced class weights
- Classification using Random Forest with out-of-bag (OOB) evaluation
- Visualization of confusion matrices

### Requirements

Make sure you have the following Python packages installed:

`pip install pandas numpy scikit-learn matplotlib`

### Usage

1. Place `csgo_round_snapshots.csv` in the same directory.  
2. Run the Python script.

### Steps Performed

### Data Preparation
- Loads first 5000 rows of data and drops missing values.  
- Converts the categorical round winner label to a numeric representation.  
- Selects relevant feature columns and the target variable.

### Normalization and Splitting
- Normalizes feature data to zero mean and unit variance.  
- Splits the dataset into 70% training and 30% test sets.

### Support Vector Classifier (SVC)
- Defines an SVC with RBF kernel and balanced class weights.  
- Performs grid search over `C` and `gamma` parameters using 5-fold cross-validation, optimizing weighted F1 score.  
- Prints cross-validation mean test scores and ranks.  
- Trains final SVC with the best found hyperparameters.  
- Prints weighted F1 score on the test set.  
- Plots normalized confusion matrix.

### Random Forest Classifier
- Performs grid search to find the optimal `max_depth` parameter.  
- Prints cross-validation mean test scores and ranks.  
- Trains Random Forest with the best `max_depth`, enables out-of-bag (OOB) scoring, and verbose output.  
- Prints training accuracy, test accuracy, and OOB score.  
- Plots normalized confusion matrix.

### Results
- Cross-validation results for hyperparameter tuning of both models printed to console.  
- Weighted F1 score for SVC on the test set printed.  
- Confusion matrices plotted for both SVC and Random Forest classifiers.  
- Training, testing, and OOB scores for Random Forest printed.
