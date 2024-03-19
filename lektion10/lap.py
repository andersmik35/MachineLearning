
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
train_df = pd.read_csv('bank_loan_data.csv')

# Dropping the 'Loan_ID' column as it's not useful for prediction
train_df = train_df.drop(columns=['Loan_ID'])

# Encoding categorical features and imputing missing values
train_df_encoded = pd.get_dummies(train_df, drop_first=True)

# Splitting the dataset into features (X) and target variable (y)
X = train_df_encoded.drop(columns='Loan_Status_Y')
y = train_df_encoded['Loan_Status_Y']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Imputing missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining the MLPClassifier model
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, activation='relu', random_state=0)

# Training the MLPClassifier model
mlp.fit(X_train, y_train)

# Predicting the target values for the test set
y_pred = mlp.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")



# Generate predictions
y_pred = mlp.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Approved', 'Approved'], yticklabels=['Not Approved', 'Approved'])
plt.title('Confusion Matrix of Loan Approval Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
