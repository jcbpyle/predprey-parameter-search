# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:03:03 2017

@author: James
"""
import random
from sklearn import preprocessing
from neural_network import Neural_Net
try:
    import cPickle as pickle
except:
    import pickle


random.seed(30051995)
print("loading neural network from file")
NN = pickle.load(open("NN.p","rb"))
print("loaded neural network")
mmscalerX = pickle.load(open("scaleX.p","rb"))
print("loaded x scaler",mmscalerX)
mmscalerY = pickle.load(open("scaleY.p","rb"))
print("loaded y scaler",mmscalerY)
X = pickle.load(open("X.p","rb"))
Y = pickle.load(open("Y.p","rb"))
mmscalerX.fit(X)
mmscalerY.fit(Y)
print("generating new params")

def gen_params():
    new = [0]*8
    new[0] = int(random.uniform(0, 5000))
    new[1] = int(random.uniform(0, 5000))
    new[2] = int(random.uniform(0, 5000))
    new[3] = random.uniform(0, 0.25)
    new[4] = random.uniform(0, 0.25)
    new[5] = int(random.uniform(0, 200))
    new[6] = int(random.uniform(0, 200))
    new[7] = int(random.uniform(0, 200))
    return new

print("test correct weights",NN.weights)
#test_params = gen_params()
test_params=[[557,380,2200,0.131234,0.025769,25,40,38],[2244,802,1876,0.132587,0.010532,119,19,46],[1391,2568,2927,0.159585,0.060718,31,18,31]]
print("test",test_params)
tp = mmscalerX.transform(test_params)
print("scaled test",tp)
prediction = NN.forward(test_params)
#prediction = NN.forward(testX[0])
#print("test prediction for",mmscalerX.inverse_transform([testX[0]]))
print("prediction",prediction)
print("scaled prediction",mmscalerY.inverse_transform([prediction]))