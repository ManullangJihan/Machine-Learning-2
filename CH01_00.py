#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:51:14 2020

@author: hanjiya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style
import pickle

style.use("ggplot")

data = pd.read_csv('student-mat.csv', sep=";")
predict = "G3"
data = data[["G1", "G2", "absences","failures", "studytime","G3"]]
data = shuffle(data)

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print('Accuracy :' + str(acc))
    
    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)
    
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)

print("------------------------")
print('Coefficients : \n', linear.coef_)
print('Intercept : \n', linear.intercept_)
print("------------------------")

predicted = linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])

p1 = "G1"
plt.scatter(data[p1], data[predict])
plt.xlabel(p1)
plt.ylabel('final grade')
plt.show()

    
    
    
    
    