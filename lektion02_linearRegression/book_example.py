import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Download and prepare data
data_root = 'https://github.com/ageron/data/raw/main/'
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = lifesat[['GDP per capita (USD)']].values
y = lifesat[['Life satisfaction']].values
print(lifesat)

# Visualize the data
lifesat.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction", grid=True)
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(x, y)

# Make a prediciton for Cyprus
X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(X_new))  # outputs [[6.25984414]]


