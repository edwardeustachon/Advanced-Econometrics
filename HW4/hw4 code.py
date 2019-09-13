import numpy as np #general library, will come in handy later
import pandas as pd #another nice library for storing matrices, it rely's on numpy

import matplotlib.pyplot as plt #this library is for graphing things
from sklearn.linear_model import Lasso #These libraries have the necessary models
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures 

df = pd.read_csv('C:/Users/eddie/Desktop/School/ECO 348K Advanced Econometrics/Homework\HW4\credit.csv')
#make dummy variables of characteristics
df.insert(2,"Female", df.Gender.map({"Male": 0, "Female": 1}), allow_duplicates=True)
df.insert(2,"MarriedD", df.Married.map({"No": 0, "Yes": 1}), allow_duplicates=True)
df.insert(2,"StudentD", df.Student.map({"No": 0, "Yes": 1}), allow_duplicates=True)
df.insert(2,"Asian", df.Ethnicity.map({"Caucasian": 0, "African American": 0, "Asian": 1}), allow_duplicates=True)
df.insert(2,"Black", df.Ethnicity.map({"Caucasian": 0, "African American": 1, "Asian": 0}), allow_duplicates=True)
#drop characteristic columns with string-types
df.drop(columns=['Gender', 'Married', 'Student', 'Ethnicity'], axis=0, inplace=True)
#create interaction terms
for x in ['Income', 'Black', 'Asian', 'StudentD', 'MarriedD', 'Female', 'Limit', 'Rating', 'Cards', 'Age', 'Education']:
    for y in ['Income', 'Black', 'Asian', 'StudentD', 'MarriedD', 'Female', 'Limit', 'Rating', 'Cards', 'Age', 'Education']:
        df.insert(12, x+"*"+y, df[x]*df[y], allow_duplicates=True)
#create x's and y's
X=df.loc[:, 'Income':'Income*Income']
y=df.loc[:, 'Balance':]
#fit lasso to our dataset
lassoreg = Lasso(alpha=1)
lassoreg.fit(X, y)
ypred=lassoreg.predict(X)
MSE = mean_squared_error(y['Balance'], ypred)
kfoldMSE = []
"""
for i in range(5):
    #compute start/end of fold
    start_index = int((400/5)*i)
    end_index = int((400/5)*(i+1))

    
    #partition data
    X_test =  X[start_index:end_index]
    y_test = y[start_index:end_index]
    
    X_train = np.concatenate( (X[0:start_index],X[end_index:]) )
    y_train = np.concatenate( (y[0:start_index],y[end_index:]) )
          
    #estimate model
    lasso = Lasso(alpha=.5)
    lasso.fit(X_train,y_train)
    yfitted=lasso.predict(X_test)
    MSEi = mean_squared_error(y_test, yfitted)
    kfoldMSE.append(MSEi)
kfoldSum = sum(kfoldMSE)
CVn = kfoldSum/5
"""
for i in range(10, 1000, 10):
    lassogrid = Lasso(alpha=i)
    lassogrid.fit(X,y)
    
#Specify a grid of lambda values to optimize over.
lambda_values = 10*np.array(range(100))

lass_cv = LassoCV(cv=5, alphas = lambda_values)
lass_cv = lass_cv.fit(X, y)

#iteratively fit model and create a graph
CVval = []
lambda_values = 10*np.array(range(100))

for lamb in lambda_values:
    l = Lasso(alpha=lamb)
    l.fit(X, y)
    
print(intercepts)
    
#graph result
plt.plot(lambda_values, intercepts)
