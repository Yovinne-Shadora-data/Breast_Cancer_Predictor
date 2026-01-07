import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('breast_cancer.csv')

# Select only the 5 features used for the model
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
X = data[features]

# Convert target labels to integers: B=0 (Benign), M=1 (Malignant)
y = data['diagnosis'].map({'B': 0, 'M': 1})

# Load saved model and scaler
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale test features
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Breast Cancer Model Accuracy: {accuracy*100:.2f}%')

# Save plot
plt.savefig('breast_cancer_accuracy_plot.png')
print("Confusion matrix plot saved as 'breast_cancer_accuracy_plot.png'")

# Close plot
plt.close()
