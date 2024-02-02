import matplotlib.pyplot as plt
import numpy as np

X = 2 * np.random.rand(100, 1)
print(X)
y = 4 + 3 * X + np.random.randn(100, 1)
print(y)
plt.plot(X,y, "b.")
plt.axis([0,2,0,15])
plt.plot()
plt.show()