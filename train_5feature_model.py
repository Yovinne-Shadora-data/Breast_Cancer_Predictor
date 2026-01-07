import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('breast_cancer.csv')
features = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean']
X = data[features]
y = data['diagnosis'].map({'B':0,'M':1})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)

with open('logistic_model.pkl','wb') as f:
    pickle.dump(model,f)
with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)

print("? 5-feature model and scaler saved successfully.")
