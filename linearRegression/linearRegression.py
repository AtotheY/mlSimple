#Anthony Sistilli
#Example using Scikit learn with Linear regression on a random dataset from the internet
#Following tutorial from: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
#Importing relevant libraries
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import warnings

#Supress random OSX warning - solution fromhttps://github.com/scipy/scipy/issues/5998
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
#Loading the dataset in - data.csv which is height(inches) vs weight(pounds)
dataset = np.genfromtxt("data.csv", delimiter=",",names=True, dtype = float)

#Getting labels
labels = dataset.dtype.names

#Getting a 20-80 split for test-train data
split = -((len(dataset[labels[0]]))/5)

#Creating the test and training sets for both x and y
x = dataset[labels[0]].reshape(len(dataset[labels[0]]),1)
x_train = x[:split]
x_test = x[split:]

y = dataset[labels[1]].reshape(len(dataset[labels[1]]),1)
y_train = y[:split]
y_test = y[split:]

#Creating the linear regression model object
reg = linear_model.LinearRegression()

#Fitting the model on the training data
reg.fit(x_train, y_train)

#Getting the coefficients that were produced for y = mx+b
print "Coefficients:",reg.coef_

#Calculating the mse (Mean Squared Error)
print "MSE:",np.mean((reg.predict(x_test) - y_test) ** 2)

#Calculatig the variance score of the model
print "Variance:",reg.score(x_test,y_test)

#Plotting the output
# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, reg.predict(x_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
