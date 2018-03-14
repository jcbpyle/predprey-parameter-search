# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:57:50 2018

@author: James
"""

import random
import sys
from math import exp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
    
#Set random seed
seed = random.randrange(sys.maxsize)
print("seed:",seed)

X = pickle.load(open("X.p","rb"))
Y = pickle.load(open("Y3.p","rb"))

s0 = preprocessing.MinMaxScaler()
X = s0.fit_transform(X,Y)

X, Y = zip(*random.sample(list(zip(X,Y)), 250000))

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.05)

trainY1 = []
trainY2 = []
trainY3 = []
testY1 = []
testY2 = []
testY3 = []
for a in trainY:
    trainY1.append(a[0])
    trainY2.append(a[1])
    trainY3.append(a[2])
for b in testY:
    testY1.append(b[0])
    testY2.append(b[1])
    testY3.append(b[2])
    
trainY1 = np.asarray(trainY1)
trainY2 = np.asarray(trainY2)
trainY3 = np.asarray(trainY3)
    
#s1 = preprocessing.MinMaxScaler()
#s2 = preprocessing.MinMaxScaler()
#s3 = preprocessing.MinMaxScaler()

#trainY1 = trainY1.reshape(-1,1)

#trainX = s1.fit_transform(trainX)
#trainY1 = s2.fit(trainY1)
#trainY1 = s2.transform(trainY1)
#trainY2 = s3.fit_transform([trainY2])


e = exp(-8)

nn1 = MLPRegressor(activation='relu', solver='adam', alpha=0.0001, max_iter=500, epsilon=e)
nn2 = MLPRegressor(activation='relu', solver='adam', alpha=0.0001, max_iter=500, epsilon=e)
nn3 = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, max_iter=500, epsilon=e)

nn1.fit(trainX,trainY1)
nn2.fit(trainX,trainY2)
nn3.fit(trainX,trainY3)
#nn1.fit(trainX,trainY1[0])
#nn2.fit(trainX,trainY2[0])
#nn3.fit(trainX,trainY3)

#testX=s1.transform(testX)
#testY1 = s2.transform(testY1)
#testY2 = s3.transform([testY2])

print("scores",nn1.score(testX,testY1),nn2.score(testX,testY2),nn3.score(testX,testY3))
#print("scores",nn1.score(testX,testY1[0]),nn2.score(testX,testY2[0]),nn3.score(testX,testY3))

test_params=[[557,380,2200,0.131234,0.025769,25,40,38],[2244,802,1876,0.132587,0.010532,119,19,46],[1391,2568,2927,0.159585,0.060718,31,18,31]]
print("test",test_params)

p1 = nn1.predict(test_params)
p2 = nn2.predict(test_params)
p3 = nn3.predict(test_params)

print("prediction1",p1)
print("prediction2",p2)
print("prediction3",p3)
print("real results:",[[56,87.238998,1],[57,171.455994,1],[54,375.036987,1]])