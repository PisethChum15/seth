# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

# Load the data
data = pd.read_csv("../data/Iris.csv")

# Checking the first few rows of the dataset
print(data.head())

# Get the class distribution
print(data['Species'].value_counts())

# Prepare features and labels
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model with increased iterations
model = LogisticRegression(max_iter=200)  # Increased max_iter
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Make a prediction for new data
print(model.predict(np.array([[2, 3, 4, 5]])))

# Save the model using pickle
with open("flower-v1.pkl", "wb") as f:
    pickle.dump(model, f)

# Load the model using pickle
with open("flower-v1.pkl", "rb") as f:
    model_pk = pickle.load(f)

# Make a prediction using the loaded model
print(model_pk.predict(np.array([[2, 3, 4, 5]])))

# Save the model using joblib
joblib.dump(model, "flower-v1.jl")

# Load the model using joblib (note the correct extension)
model_jl = joblib.load("flower-v1.jl")

# Make a prediction using the joblib loaded model
print(model_jl.predict(np.array([[2, 3, 4, 5]])))
