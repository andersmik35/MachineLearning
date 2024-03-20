import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Load and preprocess the dataset
df = pd.read_csv('StudentAbsenceGrade.csv')
df[['student_absence', 'grade']] = df['student_absence;grade'].str.split(';', expand=True)

# Convert columns to numeric, assuming they might be read as strings
df['student_absence'] = pd.to_numeric(df['student_absence'], errors='coerce')
df['grade'] = pd.to_numeric(df['grade'], errors='coerce')

# Preprocessing: Replace specific grades
df['grade'] = df['grade'].replace({0: 2, 4: 7, 10: 12})

# Categorize grades into 'low', 'middle', 'high'
bins = [0, 6, 10, 20]  # Adjust bins according to your grade distribution
labels = ['low', 'middle', 'high']
df['grade_category'] = pd.cut(df['grade'], bins=bins, labels=labels, right=False)

# Check for any missing or invalid data after transformations
df.dropna(subset=['student_absence', 'grade_category'], inplace=True)

print(df)
X = np.array(df['student_absence']).reshape(-1, 1)
y = df['grade_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

# Define and train the MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=0)
mlp.fit(X_train, y_train)

# Make predictions on the test set
predictions = mlp.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions, labels=['low', 'middle', 'high'])
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['low', 'middle', 'high'], yticklabels=['low', 'middle', 'high'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print the classification report for more details
print(classification_report(y_test, predictions, labels=['low', 'middle', 'high']))
