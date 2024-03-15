import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
data = pd.read_csv('titanic_train_500_age_passengerclass.csv', sep=',', header=0)

# Replace missing values
data["Pclass"].fillna(3, inplace=True)  # Unknown class set to 3
data["Age"].fillna(27, inplace=True)    # Unknown age set to 27
data["Survived"].fillna(1, inplace=True)  # Unknown survival set to survived

# Prepare the labels
yvalues = pd.DataFrame(data["Survived"].copy(), dtype=int)

# Visualize the data
plt.figure()
plt.scatter(data["Age"].values, data["Pclass"].values, color='black', s=20)
plt.xlabel("Age")
plt.ylabel("Passenger Class")
plt.title("Scatter Plot of Age vs Passenger Class")
plt.show()

# Remove unnecessary columns
data.drop(['Survived', 'PassengerId'], axis=1, inplace=True)

# Split the dataset into training and test sets
xtrain = data.head(400)
xtest = data.tail(100)
ytrain = yvalues.head(400)
ytest = yvalues.tail(100)

# Scale the features
scaler = StandardScaler()
scaler.fit(xtrain)  # Fit only to the training data

X_train = scaler.transform(xtrain)
X_test = scaler.transform(xtest)

# Initialize and train the neural network
mlp = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=1000, activation="relu")
mlp.fit(X_train, ytrain.values.ravel())

# Make predictions and evaluate the model
predictions = mlp.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(ytest, predictions))
print("\nClassification Report:")
print(classification_report(ytest, predictions))

# Calculate accuracy
accuracy = sum(predictions == ytest['Survived'].values) / len(ytest)
print(f"Accuracy: {accuracy * 100:.2f}%")



