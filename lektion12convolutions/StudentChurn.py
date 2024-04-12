# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:41:07 2021
@author: sila
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data using semicolon as delimiter
data = pd.read_csv('studentchurn.csv', delimiter=';')

# Handling missing values by filling them with the most frequent value in the StudyGroup column
most_frequent_studygroup = data['StudyGroup'].mode()[0]
data['StudyGroup'].fillna(most_frequent_studygroup, inplace=True)

# Convert 'StudyGroup' to integer type since it is binary
data['StudyGroup'] = data['StudyGroup'].astype(int)


# Replace string labels with numerical codes
data['Churn'] = data['Churn'].replace(['Completed', 'Stopped'], [1, 0])
data['Line'] = data['Line'].replace(['HTX', 'HF', 'STX', 'HHX', 'EUX'], [1, 2, 3, 4, 5])

# Separate features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn'].astype(int)  # Ensure target is integer

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scaling numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]))

# Random Forest model
rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(X_train_scaled, y_train)
y_pred_rf = rf_classifier.predict(X_test_scaled)

# Model evaluation
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Calculate accuracy
accuracy_rf = np.mean(y_pred_rf == y_test)
print(f"Random Forest Accuracy: {accuracy_rf:.2%}")













# x = data[ "Line" ]
# y = data[ "Age" ]
# plt.figure()
# plt.scatter(x.values, y.values, color = 'black' , s = 20 )
# plt.show()
#
# x = data[ "Grade" ]
# y = data[ "Age" ]
# plt.figure()
# plt.scatter(x.values, y.values, color = 'black' , s = 20 )
# plt.show()
#
# x = data[ "Line" ]
# y = data[ "Grade" ]
# plt.figure()
# plt.scatter(x.values, y.values, color = 'black' , s = 20 )
# plt.show()
#
# x = data[ "Grade" ]
# y = data[ "Churn" ]
# plt.figure()
# plt.scatter(x.values, y.values, color = 'black' , s = 20 )
# plt.show()