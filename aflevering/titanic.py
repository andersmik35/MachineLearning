# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Data preprocessing
data = pd.read_csv('titanic_800.csv')
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Sex'] = data['Sex'].replace(['female', 'male'], [1, 0])
embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
data = pd.concat([data, embarked_dummies], axis=1)
data.drop('Embarked', axis=1, inplace=True)

X = data.drop('Survived', axis=1)
y = data['Survived']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
y_pred_rf = rf_classifier.predict(X_test_scaled)

# MLP
mlp_classifier = MLPClassifier(random_state=42, max_iter=300)
mlp_classifier.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_classifier.predict(X_test_scaled)

# Evaluating RF
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Evaluating MLP
print("MLP Classification Report:\n", classification_report(y_test, y_pred_mlp))
print("MLP Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))

# Accuracy
accuracy_rf = np.mean(y_pred_rf == y_test)
print(f"Random Forest Accuracy: {accuracy_rf:.2%}")

accuracy_mlp = np.mean(y_pred_mlp == y_test)
print(f"MLP Accuracy: {accuracy_mlp:.2%}")



