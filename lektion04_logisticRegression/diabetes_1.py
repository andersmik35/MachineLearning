import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generate synthetic data
np.random.seed(0)
n_samples = 100
BMI = np.random.normal(25, 5, n_samples).reshape(-1, 1)
# Generating random diabetes status based on BMI
prob_diabetes = 1 / (1 + np.exp(-(-5 + 0.15 * BMI)))
diabetes = np.random.binomial(1, prob_diabetes)

# Fit logistic regression model
model = LogisticRegression()
model.fit(BMI, diabetes)

# Generate predictions with the model
BMI_values = np.linspace(10, 40, 300).reshape(-1, 1)
diabetes_probability = model.predict_proba(BMI_values)[:, 1]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(BMI, diabetes, color='black', marker='.', label='Data points')
plt.plot(BMI_values, diabetes_probability, color='blue', label='Logistic Regression')
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Probability of Diabetes')
plt.title('Correlation between BMI and Diabetes')
plt.legend()
plt.grid(True)
plt.show()
