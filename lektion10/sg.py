import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('StudentGrades.csv')
df.head()
df.tail()
df.describe()


plt.scatter(x = df.study_hours, y = df.student_marks)
plt.xlabel("Student Study Hours")
plt.ylabel("Student Marks")
plt.title("Scatter Plot of Student Study Hours vs Student Marks")
plt.show()

df.isnull().sum()
df2 = df.fillna(df.mean())
df2.isnull().sum()
df2.head()

X = df2.drop("student_marks", axis = "columns")
y = df2.drop("study_hours", axis = "columns")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=51 )
print ("Shape of X_train = ", X_train.shape)
print ("Shape of y_train = ", y_train.shape)
print ("Shape of X_test = ", X_test.shape)
print ("Shape of y_test = ", y_test.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train , y_train)
y_pred = lr.predict(X_test)


plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test)
plt.plot(X_train, lr.predict(X_train), color = "r" )


# I would like to make 3 clusters for this dataset

combined = np.column_stack((X,y))
print(combined)

k = 3
kmeans = KMeans(n_clusters=k, random_state=0).fit(combined)
labels = kmeans.labels_
clusters = kmeans.cluster_centers_
print(clusters)

from matplotlib.colors import ListedColormap
cmap_bold = [ListedColormap(['#FF0000', '#00FF00', '#0000FF']),
             ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00']),
             ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00','#00FFFF'])]

plt.scatter(X, y, c=labels, edgecolor='black', cmap=cmap_bold[0], s=20)

plt.show()




