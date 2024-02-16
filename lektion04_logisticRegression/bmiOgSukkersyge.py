#get me utf 8
# -*- coding: utf-8 -*-
import numpy
from sklearn import linear_model
import matplotlib.pyplot as plt


#Reshaped for Logistic function.
#X numbers are the the BMI
#Dårlige data tho
X = numpy.array([13,18.5,23,25,25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]).reshape(-1,1)
#y numberrs if 0 then false if 1 then true, so 1 = yes it is diabetes.
y = numpy.array([0, 0, 0, 0,1,1,1,1,1,1,1,1,1,1,1,1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

#Det her printer bare med en given værdi om personen har diabetes eller ej.
#predict if person has diabetes where the BMI is 24.4:
# predicted = logr.predict(numpy.array([24.4]).reshape(-1,1))
# print(predicted)


#Herefter kan man også plotte hvis man vil
X_test = numpy.linspace(10, 90, 300)[:, numpy.newaxis]
probabilities = logr.predict_proba(X_test)[:,1] # Probability of class 1

# Plot the logistic regression curve
plt.figure()
plt.plot(X_test, probabilities, color='r', linewidth=3)

# Plot the data points
plt.scatter(X, y, color='black')

# Set labels and title
plt.xlabel('BMI')
plt.ylabel('Probability of Diabetes')
plt.title('Logistic Regression')

plt.show()
