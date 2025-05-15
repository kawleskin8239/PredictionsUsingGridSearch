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
The data used in this project comes from the Kaggle dataset:
**[Machine Predictive Maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)**

It includes various sensor readings and categorical variables related to machine operation and failure status.

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

#### Preprocessing
- Loads data and drops missing values.  
- Drops initial irrelevant columns.  
- Converts object columns to categorical and encodes them as numeric codes.  
- Prints a dictionary showing category encodings for categorical features.  
- Separates features and target labels.

#### Train/Test Split
- Splits data into 70% training and 30% test sets.

#### Hyperparameter Tuning
- Uses GridSearchCV to find the optimal `max_depth` for the Decision Tree.  
- Prints cross-validation mean test scores and ranks.

#### Training and Evaluation
- Trains the Decision Tree with the best `max_depth` and balanced class weights.  
- Displays normalized confusion matrix.  
- Visualizes the decision tree structure limited to a depth of 2 for clarity.

### Results
- Dictionary of categorical variable encodings is printed to console:
  `{'Type': {0: 'H', 1: 'L', 2: 'M'}, 'Failure Type': {0: 'Heat Dissipation Failure', 1: 'No Failure', 2: 'Overstrain Failure', 3: 'Power Failure', 4: 'Random Failures', 5: 'Tool Wear Failure'}}`  
- Scores for different tree depths are printed.
  
<img width="329" alt="image" src="https://github.com/user-attachments/assets/4d54221c-d3ae-4819-b759-4b787317a34f" />

- Confusion matrix is plotted for opitmal decision tree
  
<img width="375" alt="image" src="https://github.com/user-attachments/assets/48da1b89-1305-45ad-8c12-b511427e9c78" />

- Decision tree visualization plot (depth limited to 2).
  
<img width="758" alt="image" src="https://github.com/user-attachments/assets/9045ad30-2eb4-412d-92fa-096b03ee48a2" />

## 3. CSGO Round Prediction
This project uses machine learning to predict the outcome of CSGO rounds based on game state snapshots. It applies Support Vector Classifier (SVC) and Random Forest models with hyperparameter tuning via GridSearchCV to optimize predictive performance.

### Dataset
The data used in this project comes from the Kaggle dataset:
**[CS:GO Round Winner CLassification](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification)**

It contains game state features such as time left, scores, and player health, along with the round winner label.

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

#### Data Preparation
- Loads first 5000 rows of data and drops missing values.  
- Converts the categorical round winner label to a numeric representation.  
- Selects relevant feature columns and the target variable.

#### Normalization and Splitting
- Normalizes feature data to zero mean and unit variance.  
- Splits the dataset into 70% training and 30% test sets.

#### Support Vector Classifier (SVC)
- Defines an SVC with RBF kernel and balanced class weights.  
- Performs grid search over `C` and `gamma` parameters using 5-fold cross-validation, optimizing weighted F1 score.  
- Prints cross-validation mean test scores and ranks.  
- Trains final SVC with the best found hyperparameters.  
- Prints weighted F1 score on the test set.  
- Plots normalized confusion matrix.

#### Random Forest Classifier
- Performs grid search to find the optimal `max_depth` parameter.  
- Prints cross-validation mean test scores and ranks.  
- Trains Random Forest with the best `max_depth`, enables out-of-bag (OOB) scoring, and verbose output.  
- Prints training accuracy, test accuracy, and OOB score.  
- Plots normalized confusion matrix.

### Results
- Results for the grid search to find the optimal Support Vector Classifier are printed to the console
  
  <img width="371" alt="image" src="https://github.com/user-attachments/assets/8f2ae540-f22a-410f-96ad-f22ad5477ebb" />

  The score on the test set for the SVC with optimal parmeters is printed at the bottom
  
  <img width="82" alt="image" src="https://github.com/user-attachments/assets/66e3c753-eb43-4aa6-8741-5878a707168b" />

- Confusion Matrix is plotted for the optimal SVC
  
  <img width="386" alt="image" src="https://github.com/user-attachments/assets/08e4a02f-61ff-4f33-b153-0f6a85ca63a3" />

- Results for the grid search to find the optimal Random Forest is printed to the console
  
  <img width="325" alt="image" src="https://github.com/user-attachments/assets/f54b2f2d-fade-4cea-b73f-c1c1bfb3b005" />

- Train, Test, and OOB score for the optimal Random Forest are printed to the console
  
  <img width="126" alt="image" src="https://github.com/user-attachments/assets/496b4ef2-5462-4bf8-9a56-2c18a22cbcc8" />

  <img width="128" alt="image" src="https://github.com/user-attachments/assets/4544189b-c69a-49a6-8734-80c8d93a6238" />

- Confusion Matris is plotted for the optimal Random Forest
  
  <img width="371" alt="image" src="https://github.com/user-attachments/assets/e2757f13-aefc-4211-9385-8b63b0ea71c1" />


