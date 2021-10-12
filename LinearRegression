'''
Linear regression is used when predicting a numeric probability predicition:
1)What will these stocks return after one year?
2)What is the chance this person will graduate from our college?
3)What is the probability this person will be able to pay off his loan?

Logistic regression is generally a yes or no question such as:
1)Does this person have cancer?
2)Will this person graduate?
3)Will this investment outperform the SP500?
'''

ESTIMATED REGRESSION MODEL:
The sample regression line provides an estimate of the population regression line.

y =  b0 + bx

y =  Estimated (or predicted value)
b0 = Estimate the regression intercept
bx = Independent variable

Regression is the attempt to explain the variation in a dependent variable using the variation in the independent variables.
Simple regression fits a straigt line to the data.

Dependent variable is Y; indepedent variable is X.


#IMPOPRT LIBRARY USING PYTHON:
import numpy as np
import pandas as pd
import matplotlit.pyplot as plt

dataset=pd.read_csv("tips.csv")
x=dataset.iloc[: , :-1].values
y=dataset.iloc[: , :-1].values

# Fitting Linear Regression:
regressor = LinearRegression()
regressor.fit(x,y)
regressor.coef_
regressor.intercept_


#PLOT:
x_mean = [np.mean(x) for i in x]
y_mean = [np.mean(y) for i in y]
plt.scatter(x,y)
plt.plot(x, regressor.predict(x), color ="red")
plt.plot(x, y_mean)
plt.title("Tip vs Bill")
plt.xlabel("Bill in $")

R2 (r-squared) represents a number between 0 and 1. The higher the value the higher the linear regression.
It is often referred to as a percentage. 

lm=sf.ols(formula = "Tip - Bill", data=dataset).fit()

#PRINT FUNCTION:
lm.params
print(lm.summary())
