# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:30:13 2018

@author: James
"""

import random
import os
import sys
from math import exp
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
    
    
    
    
    
PATH = os.getcwd()
DIRECTORY = "/training_results/"    

#Set random seed
seed = random.randrange(sys.maxsize)
print("seed:",seed)

X = pickle.load(open("X.p","rb"))
Y = pickle.load(open("Y3.p","rb"))

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

e = exp(-8)

#nn1 = MLPRegressor(activation='relu', solver='adam', alpha=0.0001, max_iter=2000, epsilon=e)
#nn2 = MLPRegressor(activation='relu', solver='adam', alpha=0.0001, max_iter=2000, epsilon=e)
#nn3 = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, max_iter=2000, epsilon=e)
#
#nn1.fit(trainX,trainY1)
#nn2.fit(trainX,trainY2)
#nn3.fit(trainX,trainY3)

print("scores",nn1.score(testX,testY1),nn2.score(testX,testY2),nn3.score(testX,testY3))

test_params=[[557,380,2200,0.131234,0.025769,25,40,38],[2244,802,1876,0.132587,0.010532,119,19,46],[1391,2568,2927,0.159585,0.060718,31,18,31]]
print("test",test_params)

#p1 = nn1.predict(test_params)
#p2 = nn2.predict(test_params)
#p3 = nn3.predict(test_params)
#
#print("prediction1",p1)
#print("prediction2",p2)
#print("prediction3",p3)
print("real results:",[[56,87.238998,1],[57,171.455994,1],[54,375.036987,1]])
#confusion_matrix(testY,p1)
#classification_report(testY,p1)

def run_tests():
    count=0
    e = exp(-8)
    test_params = [[557,380,2200,0.131234,0.025769,25,40,38],[2244,802,1876,0.132587,0.010532,119,19,46],[1391,2568,2927,0.159585,0.060718,31,18,31]]
    real_results = [[56,87.238998,1],[57,171.455994,1],[54,375.036987,1]]
    scaler=None
    for d in [0.001,0.01,0.1,0.25,0.5,0.75,0.8,0.85,0.9,0.95,0.99]:
        for a in ["relu","identity","logistic","tanh"]:
            for sol in ["lbfgs","sgd","adam"]:
                for sc in ["std","mm","ma","no","ro"]:
                    for hl in [(0,), (5,3), (30,30,30), (100,)]:
                        for lr in ["constant","invscaling","adaptive"]:
                            for t in ["y","n"]:
                                count+=1
                                if(count<2):
                                    sv = open(PATH+DIRECTORY+str(d)+"_"+a+"_"+sol+"_"+sc+"_"+str(hl[0])+"_"+lr+".txt","w")
                                    sv.write("seed: "+str(seed))
                                    if(t=="n"):
                                        
                                        trainX, testX, trainY, testY = train_test_split(X,Y, test_size=d)
                                        
                                        if(sc=="std"):
                                            scaler=StandardScaler()
                                        elif(sc=="mm"):
                                            scaler=MinMaxScaler()
                                        elif(sc=="ma"):
                                            scaler=MaxAbsScaler()
                                        elif(sc=="no"):
                                            scaler=Normalizer()
                                        else:
                                            scaler=RobustScaler()
                                            
                                        scaler.fit(trainX)
                                        trainX = scaler.transform(trainX)
                                        testX = scaler.transform(testX)
                                        
                                        if(hl[0]==0):
                                            nn1 = MLPRegressor(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e)
                                            nn2 = MLPClassifier(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e)
                                        else:
                                            nn1 = MLPRegressor(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e, hidden_layer_sizes=hl)
                                            nn2 = MLPClassifier(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e, hidden_layer_sizes=hl)
                                        
                                        nn1.fit(trainX,trainY)
                                        nn2.fit(trainX,trainY)
                                        
                                        sv.write("score1: "+str(nn1.score(testX,testY)))
                                        sv.write("score1: "+str(nn2.score(testX,testY)))
                                        
#                                        p1 = nn1.predict(test_params)
#                                        p2 = nn2.predict(test_params)
                                        
                                        print("scores for test_params",nn1.score(test_params, real_results),nn2.score(test_params,real_results))
                                        
#                                        sv.write("conf_matrix1: "+str(confusion_matrix(testY,p1)))
#                                        sv.write("conf_matrix2: "+str(confusion_matrix(testY,p2)))
#                                        sv.write("classification_report1: "+str(classification_report(testY,p1)))
#                                        sv.write("classification_report2: "+str(classification_report(testY,p2)))
                                        sv.close()
                                    else:
                                        trainX, testX, trainY, testY = train_test_split(X,Y, test_size=d)
                                        
                                        if(sc=="std"):
                                            scaler=StandardScaler()
                                        elif(sc=="mm"):
                                            scaler=MinMaxScaler()
                                        elif(sc=="ma"):
                                            scaler=MaxAbsScaler()
                                        elif(sc=="no"):
                                            scaler=Normalizer()
                                        else:
                                            scaler=RobustScaler()
                                        
                                        scaler.fit(trainX)
                                        trainX = scaler.transform(trainX)
                                        testX = scaler.transform(testX)
                                        
                                        trainY1 = []
                                        trainY2 = []
                                        trainY3 = []
                                        testY1 = []
                                        testY2 = []
                                        testY3 = []
                                        for i in trainY:
                                            trainY1.append(i[0])
                                            trainY2.append(i[1])
                                            trainY3.append(i[2])
                                        for j in testY:
                                            testY1.append(j[0])
                                            testY2.append(j[1])
                                            testY3.append(j[2])
                                            
                                        trainY1 = np.asarray(trainY1)
                                        trainY2 = np.asarray(trainY2)
                                        trainY3 = np.asarray(trainY3)
                                        if(hl[0]==0):
                                            nn1 = MLPRegressor(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e)
                                            nn2 = MLPRegressor(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e)
                                            nn3 = MLPClassifier(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e)
                                        else:
                                            nn1 = MLPRegressor(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e, hidden_layer_sizes=hl)
                                            nn2 = MLPRegressor(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e, hidden_layer_sizes=hl)
                                            nn3 = MLPClassifier(activation=a, solver=sol, alpha=0.0001, max_iter=2000, epsilon=e, hidden_layer_sizes=hl)
                                        
                                        nn1.fit(trainX,trainY1)
                                        nn2.fit(trainX,trainY2)
                                        nn3.fit(trainX,trainY3)
                                        
                                        sv.write("score1: "+str(nn1.score(testX,testY1)))
                                        sv.write("score2: "+str(nn2.score(testX,testY2)))
                                        sv.write("score3: "+str(nn3.score(testX,testY3)))
                                        
                                        print("scores for test_params",nn1.score(test_params, real_results[:0]),nn2.score(test_params,real_results[:1]),nn3.score(test_params,real_results[:2]))
                                        
#                                        p1 = nn1.predict(test_params)
#                                        p2 = nn2.predict(test_params)
#                                        p3 = nn3.predict(test_params)
#                                        
#                                        sv.write("conf_matrix1: "+str(confusion_matrix(testY1,p1)))
#                                        sv.write("conf_matrix2: "+str(confusion_matrix(testY2,p2)))
#                                        sv.write("conf_matrix3: "+str(confusion_matrix(testY3,p3)))
#                                        sv.write("classification_report1: "+str(classification_report(testY1,p1)))
#                                        sv.write("classification_report2: "+str(classification_report(testY2,p2)))
#                                        sv.write("classification_report3: "+str(classification_report(testY3,p3)))
                                        sv.close()
    return

run_tests()