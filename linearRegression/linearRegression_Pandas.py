#Anthony Sistilli
#Example using Scikit learn with Linear regression on a random dataset from the internet
#Following tutorial from: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
##############################################################################
#Importing relevant libraries (This file includes usage of pandas)
import pandas
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#Loading in the features and labels from our data.csv file using pandas
height = pandas.read_csv('data.csv', usecols = [0])
weight = pandas.read_csv('data.csv', usecols = [1])

#Splitting the data - 10% test and 90% training
split = -((len(height))/10)

x_training_data = height[:split]
x_test_data = height[split:]

y_training_data = weight[:split]
y_test_data = weight[split:]

#Creating the linear regression model object
reg = linear_model.LinearRegression()

#Fitting the model on the training data
reg.fit(x_training_data, y_training_data)

#Getting the coefficients that were produced for y = mx+b
print "Coefficients:",reg.coef_

#Calculating the mse (Mean Squared Error)
print "MSE:",np.mean((reg.predict(x_test_data) - y_test_data) ** 2)

#Calculatig the variance score of the model
print "Variance:",reg.score(x_test_data,y_test_data)

#Plotting the output
# Plot outputs
plt.scatter(x_test_data, y_test_data, color='black')
plt.plot(x_test_data, reg.predict(x_test_data), color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
