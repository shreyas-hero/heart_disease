import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Print current working directory to check where script runs
print("Current working directory:", os.getcwd())

# Load dataset (make sure heart_disease_data.csv is in this folder)
data = pd.read_csv('heart_disease_data.csv')

# Separate features and target
X = data.drop(columns='target', axis=1)
y = data['target']

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2)

# Initialize and train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model to model.pkl
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ model.pkl created successfully!")