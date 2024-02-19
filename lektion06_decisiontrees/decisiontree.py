from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load moon dataset
X_moon, y_moon = make_moons(n_samples=1000, noise=0.3, random_state=42)

# Split the moon dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_moon, y_moon, test_size=0.2, random_state=42)

# Visualize the moon dataset
plt.figure()
plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon, cmap=plt.cm.RdBu, edgecolor='black', s=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Moon Dataset')

# Train a decision tree classifier on the moon dataset
tree_clf_moon = DecisionTreeClassifier(max_depth=2)
tree_clf_moon.fit(X_train, y_train)

# Plot decision boundary and decision tree lines for the moon dataset
x_min, x_max = X_moon[:, 0].min() - 0.5, X_moon[:, 0].max() + 0.5
y_min, y_max = X_moon[:, 1].min() - 0.5, X_moon[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

plt.contourf(xx, yy, tree_clf_moon.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolor='black', s=20)

# Plot decision tree lines
def plot_decision_boundary(tree, X, y, axes=[0, 7.5, -1.5, 2]):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = tree.predict(X_new).reshape(x1.shape)
    plt.contour(x1, x2, y_pred, alpha=0.8, cmap=plt.cm.RdBu)

plot_decision_boundary(tree_clf_moon, X_train, y_train)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary with Decision Tree Lines (Moon Dataset)')
plt.savefig('moon_decision_boundary_with_tree_lines.png')  # Save the plot as a PNG
plt.show()


