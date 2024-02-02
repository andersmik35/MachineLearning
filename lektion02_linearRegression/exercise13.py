# Python version
import random

import numpy as np
import matplotlib.pyplot as plt


def cost(a, b, X, y):
    # Evaluate half MSE (Mean square error)
    m = len(y)
    error = a + b * X - y
    J = np.sum(error ** 2) / (2 * m)
    return J


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

ainterval = np.arange(1, 10, 0.10)
binterval = np.arange(0.5, 5, 0.10)

low = cost(0, 0, X, y)
bestatheta = 0
bestbtheta = 0
for atheta in ainterval:
    for btheta in binterval:
        # print("xy: %f:%f:%f" % (atheta,btheta,cost(atheta,btheta, X, y)))
        if cost(atheta, btheta, X, y) < low:
            low = cost(atheta, btheta, X, y)
            bestatheta = atheta
            bestbtheta = btheta

print("a and b: %f:%f" % (bestatheta, bestbtheta))

h = bestatheta + bestbtheta * 6
print("Hypothesis: ", h)
print("MSE: %f" % (np.sum(h ** 2) / (2 * len(y))))

plt.plot(X, y, "b.")
# plot the best hypothesis and the best line
plt.plot(X, bestatheta + bestbtheta * X, "r-")

plt.show()