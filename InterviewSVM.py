'''
Support vector machine classifcation interview coding project for a fortune50:
'''

#Data imports
import pandas as pd
import numpy as np
import time

#Visualization imports
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

'''
Step1: CLEAN DATA: 
'''


start=time.time()
print("Support vector machine classification starting!")

df=r'C:\Users\scott\Desktop\exercise_40_train.csv' 
df=pd.read_csv(df)

#Drop columns that provide no numeric value:
df.drop(['x3', 'x7', 'x19',  'x24', 'x30', 'x31', 'x33', 'x39',
         'x42', 'x44', 'x49', 'x52', 'x57', 'x58', 'x60', 'x65', 
         'x67', 'x77', 'x93', 'x99'], axis = 1, inplace = True)

df.dropna(inplace=True) #delete any rows with missing values the simple way.
#A more complex way would be to inmpute the mean for 

'''
Step 2 -  BUILD MODELS AND TRAIN DATA:
'''

#Split the data set into x and y data
y_data = df['y']
#print(y_data)
x_data = df.drop('y', axis = 1)


#Split the data set into training data and test data
y_data = df['y']
x_data = df.drop('y', axis = 1)

#Split the data set into training data and test data
from sklearn.model_selection import train_test_split
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.3, random_state=42)

#Train the SVM model
from sklearn.svm import SVC
model = SVC()
model.fit(x_train_data, y_train_data)

'''
STEP 3: MAKE PREDICTIONS:
'''

#Make predictions with the model
predictions = model.predict(x_test_data)

#Performance measurement:
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
#print(classification_report(y_test_data, predictions))
#print(confusion_matrix(y_test_data, predictions))

metrics.mean_absolute_error(y_test_data, predictions)
metrics.mean_squared_error(y_test_data, predictions)
np.sqrt(metrics.mean_squared_error(y_test_data, predictions))


print(classification_report(y_test_data, predictions))
print(confusion_matrix(y_test_data, predictions))

#use model to predict probability that given y value is 1:
y_pred_proba = model.predict(x_test_data)

#calculate AUC of model
results=[]
auc = round( metrics.roc_auc_score(y_test_data, y_pred_proba), 4 ) 
print("AUC is: ", auc)
results.append(auc) #going to store results in a list


svmDF = pd.DataFrame(results)
svmDF.to_csv(r'C:\Users\Scott\Desktop\svm.csv', index=False, header=False) #index, header = false gets rid of index, header  
print("Logistic regression program finished in: ",  round(time.time()-start, 3), " seconds.")


'''
Support vector machine classification starting!
              precision    recall  f1-score   support

           0       0.97      1.00      0.99        37
           1       0.00      0.00      0.00         1

    accuracy                           0.97        38
   macro avg       0.49      0.50      0.49        38
weighted avg       0.95      0.97      0.96        38

[[37  0]
 [ 1  0]]
AUC is:  0.5
Logistic regression program finished in:  0.767  seconds.
'''
