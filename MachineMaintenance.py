import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv("predictive_maintenance.csv")
df.dropna()
df = df.iloc[:,2:]

# Identify numeric and categorical columns
num_columns = df.select_dtypes("float").columns.append(df.select_dtypes("int").columns)
cat_columns = df.select_dtypes("object").columns

# Convert object columns to categorical and print the dictionary
df[cat_columns] = df[cat_columns].astype("category")
cat_dict = {cat_columns[i]: {j: df[cat_columns[i]].cat.categories[j] for j in
range(len(df[cat_columns[i]].cat.categories))} for i
in range(len(cat_columns))}
print(cat_dict)

# Encode categorical columns
df[df.select_dtypes("category").columns] = df[df.select_dtypes("category").columns].apply(lambda x: x.cat.codes)
df.dropna(inplace=True)

# Select variables
y = df.iloc[:, -1].copy().to_numpy()
X = df.iloc[:, :-1].copy().to_numpy()

# Train/test split with 30% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Grid search to find optimal tree depth
clf = DecisionTreeClassifier(class_weight='balanced')
parameters = {"max_depth": range(2,16)}
grid_search = GridSearchCV(clf,param_grid=parameters, cv=5)
clf.fit(X_train, y_train)

# Fit and report cross-validation scores
grid_search.fit(X_train,y_train)
score_dif = pd.DataFrame(grid_search.cv_results_)
print(score_dif[['param_max_depth', 'mean_test_score', 'rank_test_score']])

# Re-train with best depth
max_depth = grid_search.best_params_["max_depth"]
clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced')
clf.fit(X_train,y_train)

# Plot confusion matrix
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

# Vissualize decision tree limited to a depth of 2
plt.figure(figsize=(30, 30))
plot_tree(clf, filled=True, feature_names=df.columns[1:], class_names=['0', '1', '2', '3', '4', '5'], max_depth=2)
plt.show()
