import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Load your dataset
data = pd.read_csv('breast_cancer.csv')  # Replace with your actual CSV file

# Separate features and target
X = data.drop('target', axis=1)  # Replace 'target' if your label column is different
y = data['target']

# Select the top 5 features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Get the selected feature names
selected_features = X.columns[selector.get_support()]
print("Top 5 selected features (ANOVA F-test):")
for feature in selected_features:
    print(feature)
