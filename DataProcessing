import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

os.getCwd()

#Normally, I usually do my dataProcessing like this using an r at the beginning:
data1=r'C:\Users\scott\OneDrive\Desktop\data1'

#However this is another mehod by putting two \\ on each \ 
data2='C:\\Users\\scott\\OneDrive\\Desktop\\data2'

df1=pd.read_csv(data1)

# df[Rows, Column]
x=data.iloc[:, :=1].values
y=data.iloc[ :, 3].values

impute = Imputer(missing_values="NaN", strategy="mean", axis=0)
impute=impute.fit(x[:,-1:3])
x[:, 1:3] = impute.fit_transform(x[:, 1:3])

# Encode categorical variables:
label_encoder_y = labelEncoder()
y=label_encoder_y.fit_transform(y)

label_encoder_x = LabelEncoder()
x[:,0 = label_encoder_x.fit_transform(x[:,0])

#CREATE DUMMIES:
onehot=OneHotEncoder(categorical_features=[0])
x=onehot.fit_transform(x).toarray()

#SPLITTING DATA: 
'''
Make model using train data.
We check how good the model is by applying in test data.
test_size: float (by default the value is set to 0.25)
train_size: (default=None) represent the porportion of th edataset to include the train split. 
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#FEATURE SCALING: Standardizing the features present in the data in a fixed range.


