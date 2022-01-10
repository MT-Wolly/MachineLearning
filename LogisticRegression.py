# -*- coding: utf-8 -*-

#importing the libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

"""
This machine learning model is used to predict if a woman has breast cancer or not.
It is a Multivariate dataset with 10 features and the model used is logistic regression.
The features are:
a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)
"""

""" Importing the dataset"""

dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

""" Splitting the dataset into the Training set and Test set"""


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

""" Training the Logistic Regression model on the Training set"""

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

""" Predicting the Test set results"""
y_pred = classifier.predict(X_test)

"""Making the Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred)
print(cm)

""" Computing the accuracy with k-Fold Cross Validation"""

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))