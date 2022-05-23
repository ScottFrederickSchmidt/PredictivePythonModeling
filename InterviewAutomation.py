'''
Automation attempt to find the best feature variables (columns x1 to x100) that have the highest significance
by taking each column individually to compare it to df['y'], the independent variable. 
For this project, I used r-squared because it is my favorite metric since it combines probability with amount of samples.
At the end, I stored the results into a dictionary with a key and value. 
Then at the very end, I print out the five highest r_squared values within the dictionary. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import time

start=time.time()
print("Starting to calculate r-squared value of each x column to y column.")

df=r'C:\Users\scott\Desktop\exercise_40_train.csv' 
df=pd.read_csv(df)

#Split the data set into training data and test data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, r2_score

y = df['y'] #Never changes as the dependent variable. 
model = LogisticRegression()
#model = LinearRegression()
r2_dict=dict()


y = df['y'] #Never changes as the dependent variable. 

for num in range(1, 101):
    try:
        col='x'+str(num) #will look through columns x1 to x100 
        x = df[col]
        r2 = r2_score(x, y) 
        r2_dict[col]=r2
        #print(col,' r2 score is: ', round(r2, 4) ) 
        time.sleep(.02) #DO NOT WANT TO CRASH MY NEW COMPUTER
    except:
        continue
        #print("error on column, column is likely NAN: ", x) 


best_dict = dict(sorted(r2_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]) #retrieves 5 highest r2 within the r2_dictionary data

for key, value in best_dict.items():
    print(key, ':', value)
        
# print( results['x2']) # dictionary is correctly working, printed r2 value of second column
print("Linear regression program finished in: ",  round(time.time()-start, 3), " seconds.")

'''
In addition, here would be an interesting attept to try all possible combinations to calculate every potential outcome using x as the variable.
This uses intertools Python permutations package.

Note: I only used less than 15 col variabes because doing all 80 would run into a memory runTime issue.
DRAFT ONLY (below):
'''

import itertools
import time
start=time.time()

col = ['x1, 'x2', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10
'x11', 'x12', 'x13', 'x14']

for L in range(0, len(stuff)+1):
    for subset in itertools.combinations(col, L):
        try:
            col='x'+str(num) #will look through columns x1 to x100 
            x = df[col]
            r2 = r2_score(x, y) 
            r2_dict[col]=r2
            #print(col,' r2 score is: ', round(r2, 4) ) 
            time.sleep(.01) #DO NOT WANT TO CRASH MY NEW COMPUTER
        except:
            continue
            #print("error on column, column is likely NAN: ", x) 

end=time.time()
print("Done calculating ", end-start, " seconds");
