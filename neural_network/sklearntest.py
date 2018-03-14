# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:51:40 2017

@author: James
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error
import random
try:
    import cPickle as pickle
except:
    import pickle

def main(s):
    random.seed(s)
    X = pickle.load(open("X.p","rb"))
    Y = pickle.load(open("Y3.p","rb"))
    
    trX,teX,trY,teY = train_test_split(X,Y)
    
    #scalerX = StandardScaler()
    scalerX = MaxAbsScaler()
    scalerX = scalerX.fit(trX)
    scalerY = MaxAbsScaler()
    scalerY = scalerY.fit(trY)
    trX = scalerX.transform(trX)
    teX = scalerX.transform(teX)
    trY = scalerY.transform(trY)
    teY = scalerY.transform(teY)
    
    #mlp = MLPRegressor(hidden_layer_sizes=(30,30,30))
    mlp = MLPRegressor(max_iter=1000, hidden_layer_sizes=(6,5), verbose=True, tol=0.000000001)
    mlp.fit(trX,trY)
    predictions = mlp.predict(teX)
    print(mean_absolute_error(teY,predictions))
    print(mean_squared_error(teY,predictions))
    print(explained_variance_score(teY,predictions))
    pickle.dump(mlp,open("mlp.p","wb"))
    #mlp = pickle.load(open("mlp.p","rb"))
    #test = gen_params()
    test =[[557,380,2200,0.131234,0.025769,25,40,38],[2244,802,1876,0.132587,0.010532,119,19,46],[1391,2568,2927,0.159585,0.060718,31,18,31]]
    test = scalerX.transform(test)
    pred = mlp.predict(test)
    print("test",test)
    print("scaled test",scalerX.inverse_transform(test))
    print("prediction",pred)
    print("Scaled prediction",scalerY.inverse_transform(pred))
    print("real results:",[[56,87.238998,1],[57,171.455994,1],[54,375.036987,1]])

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

main(74496)



































#Y0 = [[x[0]] for x in trY]
#    Y1 = [[x[1]] for x in trY]
#    tY0 = [[x[0]] for x in teY]
#    tY1 = [[x[1]] for x in teY]
#
#scalerY0 = MaxAbsScaler()
#    scalerY1 = MaxAbsScaler()
#    scalerX.fit(trX)
#    scalerY0.fit(Y0)
#    scalerY1.fit(Y1)
#    trX = scalerX.transform(trX)
#    teX = scalerX.transform(teX)
#    trY = zip(scalerY0.transform(Y0),scalerY1.transform(Y1))
#    teY = zip(scalerY0.transform(tY0),scalerY1.transform(tY1))