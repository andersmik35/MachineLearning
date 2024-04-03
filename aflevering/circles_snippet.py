import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
    plt.show()

# Generate 2D classification dataset
X, y = make_circles(n_samples=100, noise=0.05, random_state=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize SVM classifiers with different kernels
svm_linear = SVC(kernel='linear', C=1.0)
svm_poly = SVC(kernel='poly', degree=2, C=1.0)  # Adjustable degree for polynomial kernel
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)

# Train, evaluate, and plot decision boundaries for each SVM classifier
for svm, title in zip([svm_linear, svm_poly, svm_rbf],
                      ['SVM with Linear Kernel', 'SVM with Polynomial Kernel', 'SVM with RBF Kernel']):
    svm.fit(X_train, y_train)  # Train model
    y_pred = svm.predict(X_test)  # Predict on test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f'{title} - Accuracy: {accuracy:.2f}')  # Print accuracy
    plot_decision_boundaries(X_train, y_train, svm, title)  # Plot decision boundaries
