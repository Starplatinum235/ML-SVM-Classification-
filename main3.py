import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Loading the dataset
data = pd.read_csv("emails.csv")

# Droping email number
data = data.drop(columns=["Email No."])

# seperating dependent and independt variables
X = data.drop(columns=["Prediction"])
y = data["Prediction"]

# 90% dataset for training and 10% for testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# SVM classifier
model = SVC(kernel='linear', random_state=42)  # You can choose different kernels like 'rbf', 'poly', etc.

# Training the Model
model.fit(x_train, y_train)

#Predictions
predicted = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)

print(f"Accuracy: {accuracy*100}")
print(f"F1 Score: {f1*100}")
print(f"Precision: {precision*100}")
print(f"Recall: {recall*100}")

# confusion matrix
cm = confusion_matrix(y_test, predicted)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
