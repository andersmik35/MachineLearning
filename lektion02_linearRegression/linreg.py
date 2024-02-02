import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# # Generate random X and Y coordinates
# num_points = 20
# x_coordinates = np.random.rand(num_points)
# y_coordinates = np.random.rand(num_points)
#
# # Plot the points
# plt.scatter(x_coordinates, y_coordinates, color='blue')
# plt.title('Randomly Generated Points')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.grid(True)
# plt.show()


# --With new points
# Generating data
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
#
# # Plotting the points
# plt.scatter(X, y, color='blue', label='Data Points')
#
# # Adding labels and title
# plt.title('Scatter Plot of Generated Data')
# plt.xlabel('X')
# plt.ylabel('y')
#
# # Displaying the plot
# plt.legend()
# plt.grid(True)
# plt.show()


# --Now with a line through the points
# Generating data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Creating a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predicting y values using the model
y_pred = model.predict(X)
print(y_pred)

# Plotting the points
plt.scatter(X, y, color='blue', label='Data Points')

# Plotting the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

# Adding labels and title
plt.title('Scatter Plot with Regression Line')
plt.xlabel('X')
plt.ylabel('y')

# Displaying the plot
plt.legend()
plt.grid(True)
plt.show()