import numpy as np #general library, will come in handy later
import pandas as pd #another nice library for storing matrices, it rely's on numpy
import matplotlib.pyplot as plt #this library is for graphing things
from sklearn import svm #These libraries have the necessary models
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv('C:/Users/eddie/Desktop/School/ECO 348K Advanced Econometrics/Homework\HW5\Hearts_Dummy.csv')
df = df.dropna(how='any', axis=0)

#scatter plots
#plt.scatter(X['Ca'], X['Oldpeak'], c=Y['AHD_Yes'])
#plt.scatter(X['Oldpeak'], X['MaxHR'], c=Y['AHD_Yes'])
#plt.scatter(X['Ca'], X['MaxHR'], c=Y['AHD_Yes'])
#plt.xlabel('Ca')
#plt.ylabel('MaxHR')

#create subset of observations
X = df.loc[df['Thal_normal'] == 1, :'Thal_normal']
Y = df.loc[df['Thal_normal'] == 1, 'AHD_Yes':]

#create training set and test set from subset
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.5, random_state=0)

#create test observation from assignment
obs = [[0, 1, 2.5, 150, 1]]

#begin model creation, fitting, and cross-validation
model = svm.SVC()
model.fit(Xtrain, Ytrain.values.ravel())
print()
print('-----------------------------------------------------------')
print('Default parameters are C=1, kernel=RBF, and gamma=auto')
print('Training data score w/ default parameters:', model.fit(Xtrain, Ytrain.values.ravel()).score(Xtrain, Ytrain.values.ravel()))
print('Test data score w/ default parameters:', model.fit(Xtrain, Ytrain.values.ravel()).score(Xtest, Ytest.values.ravel()))
print('-----------------------------------------------------------')
parameters = [
  {'C': [0.01, .1, 1, 10, 100], 'kernel': ['linear']},
  {'C': [0.01, .1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']}]
clf = GridSearchCV(model, parameters, cv=5)
clf.fit(Xtrain, Ytrain.values.ravel())
print()
print('-----------------------------------------------------------')
print('CV parameters are:', clf.best_params_)
print('Training data w/ CV parameters:', clf.best_score_) 
print('Test data score w/ CV parameters:', clf.score(Xtest, Ytest.values.ravel()))
print('-----------------------------------------------------------')

#use test observation on cross-validated model
model = svm.SVC(C=1, kernel='linear')
model.fit(X, Y.values.ravel())
yhat = model.predict(obs)
print()
print('-----------------------------------------------------------')
print('Test observation is class:', yhat)
print('-----------------------------------------------------------')