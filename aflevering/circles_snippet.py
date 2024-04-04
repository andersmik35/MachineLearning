import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def plot_decision_boundaries(X, y, model, title, subplot):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    subplot.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    subplot.set_title(title)
    subplot.set_xlabel('Feature 1')
    subplot.set_ylabel('Feature 2')

X, y = make_circles(n_samples=100, noise=0.05, random_state=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# SVM with different kernels
svm_linear = SVC(kernel='linear', C=1.0)
svm_poly = SVC(kernel='poly', degree=2, C=1.0)
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)

# Subplots for different kernel
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Traning, etc.
for svm, title, ax in zip([svm_linear, svm_poly, svm_rbf],
                          ['SVM with Linear Kernel', 'SVM with Polynomial Kernel', 'SVM with RBF Kernel'],
                          axs.flatten()):
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{title} - Accuracy: {accuracy:.2f}')
    plot_decision_boundaries(X_train, y_train, svm, title, ax)

plt.tight_layout()
plt.show()
