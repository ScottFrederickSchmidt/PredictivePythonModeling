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

INTRODUCTION TO LOGISTIC REGRESSION:
Logistic regression should be used for "yes" or "no" questions.
Does person have heart disease? Will the person graduate?

y=B0 + B1 X
P = 0 and 1 #probability


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset:
df=pd.read_csv("Advertisement.csv")
x=df.iloc[, [2,3]].values
y=df.iloc[: 4].values

#Encoding categorical variables:
from sklearn.preprcessing import labelEncoder
label_encoder_y = labelEncoder()
y= label_encoder_y.fit_transform(y)

# Splitting dataset into training set and test set
from sklear.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.25, random_state=0)

#Feature scaling:
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


#Fitting Logistic Regressiont into training set:
from sklearn.linear_model import Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.it(x_train, y_train)

'''
Confusion Matrix Notes:

H0: Bought;         Positive
H1: Did not bought; Negative

Definition: confusion_matrix(y_true, y_pred, labels=None, simple_weight=None)
Type: Function in sklearn metrics classification module.
'''
#Confusion Matrix:
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

pd.crosstab(y_test, y_pred, rownames=["True", columns= ["Predicted"], margins=True)

#Probabilities:
prob=classifier.predicit_prob(x_test)[:, -1] > 0.5
cm2 = confusion_matrix(y_test, prob)


#PLOTS UNDERSTANDING

