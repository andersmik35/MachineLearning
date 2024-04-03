from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Generate 2D classification dataset
X, y = make_circles(n_samples=100, noise=0.05, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, title):
    # Create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Make predictions for each point in mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and margins
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')


# Initialize SVM models with different kernels
kernels = ['linear', 'poly', 'rbf']
models = {kernel: SVC(kernel=kernel) for kernel in kernels}

# Train models and evaluate
results = {}
for kernel, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    #Prints the accuracy for each kernel
    print(f'Accuracy for SVM with {kernel} kernel: {accuracy:.2f}')

    # Store results
    results[kernel] = accuracy

    # Plot decision boundaries
    plt.figure(figsize=(8, 6))
    plot_decision_boundaries(X_train, y_train, model, f'SVM with {kernel.capitalize()} Kernel')
    plt.show()

