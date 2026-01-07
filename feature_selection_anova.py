import pandas as pd 
from sklearn.feature_selection import SelectKBest, f_classif 
data = pd.read_csv('breast_cancer.csv') 
X = data.drop(['id','diagnosis'], axis=1) 
y = data['diagnosis'] 
selector = SelectKBest(score_func=f_classif, k=5) 
X_new = selector.fit_transform(X, y) 
selected_features = X.columns[selector.get_support()] 
print("Top 5 selected features (ANOVA F-test):") 
for feature in selected_features: 
    print(feature) 
