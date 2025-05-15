import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Read in data limited to first 5000 rows
df = pd.read_csv("csgo_round_snapshots.csv", nrows=5000)
df.dropna(inplace=True)

# Convert round winner to numeric
df['intWinner'] = df['round_winner'].map({'CT': 0, 'T': 1})

# Select variables
y = df['intWinner'].copy().to_numpy()
X = df[['time_left','ct_score','t_score', 'ct_health', 't_health']].copy().to_numpy()

# Normalize data
X -= np.average(X)
X /= np.std(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define and tune SVC with RBF kernel using grid search on C and gamma
reg = SVC(kernel='rbf', class_weight='balanced')
parameters = {"C": np.linspace(1, 15, num=5), "gamma": np.linspace(1, 15, num=5)}
grid_search = GridSearchCV(reg, param_grid = parameters, cv=5, scoring="f1_weighted", n_jobs=-1)
grid_search.fit(X_train,y_train)
score_dif = pd.DataFrame(grid_search.cv_results_)
print(score_dif[['param_C','param_gamma', 'mean_test_score', 'rank_test_score']])

# Fit final SVC with best C and gamma
c = grid_search.best_params_['C']
gamma = grid_search.best_params_['gamma']
clf = SVC(C=c, gamma=gamma, kernel='rbf', class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

# Plot confusion matrix for SVC
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

# Grid search on Random Forest Classifier to find best max depth
reg = RandomForestClassifier(oob_score=True)
parameters = {"max_depth": [int(x) for x in np.linspace(1, 15, num=5)]}
grid_search = GridSearchCV(reg, param_grid = parameters, cv=5)
grid_search.fit(X_train,y_train)
score_dif = pd.DataFrame(grid_search.cv_results_)
print(score_dif[['param_max_depth', 'mean_test_score', 'rank_test_score']])

# Fit final random forest with best max depth
max_depth = grid_search.best_params_['max_depth']
clf = RandomForestClassifier(max_depth=max_depth, oob_score=True, verbose=3)
clf.fit(X_train, y_train)

# Print train and test accuracy and OOB scores
print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")

# Plot confusion matrix for Random Forest
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
