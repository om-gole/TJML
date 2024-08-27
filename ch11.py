import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Simulate a financial dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load a pre-trained model (for illustration purposes, we're using a simple logistic regression model here)
pretrained_model = LogisticRegression()
pretrained_model.fit(X_train, y_train)

# Save the model weights
weights = pretrained_model.coef_

# Transfer learning: Create a new model for the actual finance problem, reusing the pre-trained weights
new_model = LogisticRegression()
new_model.coef_ = weights

# Fit the new model on the financial data (you would replace X_train, y_train with your financial dataset)
new_model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = new_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy of the transfer learned model: {accuracy:.2f}')
