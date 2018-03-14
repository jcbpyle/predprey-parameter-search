# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:14:15 2017

@author: James
"""

import numpy as np
import random
import sys
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
try:
    import cPickle as pickle
except:
    import pickle

class Neural_Net(object):
    def __init__(self, input_size=3, output_size=1, hidden_layers=0, hidden_layer_sizes=[], Lambda=0):
        if not len(hidden_layer_sizes)==hidden_layers:
            print("Specified sizes of hidden layers does not correspond to specified number of hidden layers")
            return
        #Define Hyperparameters
        print("Initialising NN internal")
        self.inputLayerSize = input_size
        self.outputLayerSize = output_size
        if not hidden_layer_sizes==[]:
            self.hiddenLayerSizes = hidden_layer_sizes
        else:
            self.hiddenLayerSizes = [0 for x in range(hidden_layers)]
        self.Lambda = Lambda
        
        #Weights (Parameters)
        self.weights = [0 for x in range(hidden_layers+1)]
        self.z = [None for x in range(len(self.weights))]
        self.a = [None for x in range(len(self.weights))]
        self.deltas = [None for x in range(len(self.weights))]
        self.grads = [None for x in range(len(self.weights))]
        print("Setting weights")
        
        for w in range(len(self.weights)):
            if hidden_layers>0:
                if w==0:
                    self.weights[w] = np.random.randn(self.inputLayerSize, self.hiddenLayerSizes[w])
                elif w==len(self.weights)-1:
                    self.weights[w] = np.random.randn(self.hiddenLayerSizes[w-1], self.outputLayerSize)
                else:
                    self.weights[w] = np.random.randn(self.hiddenLayerSizes[w-1], self.hiddenLayerSizes[w])
            else:
                self.weights[w] = np.random.randn(self.inputLayerSize, self.outputLayerSize)
        print("Initial weights set")
        return
            
    def forward(self, X):
        #Propagate inputs through network
        for v in range(len(self.z)+1):
            if v==0:
                self.z[v] = np.dot(X, self.weights[v])
            elif v==len(self.z):
                yHat = self.sigmoid(self.z[v-1])
                return yHat
            else:
                self.z[v] = np.dot(self.a[v-1], self.weights[v])
            self.a[v] = self.sigmoid(self.z[v])            
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, matrix
        #print(z)
        return 1/(1+np.exp(-z))
        
    def sigmoidPrime(self, z):
        #Derivative of Sigmoid Function
        return np.exp(-z)/((1+np.exp(-z))**2)
        
    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum(sum((y-self.yHat)**2)/X.shape[0])
#        print("J",J)
#        part = np.empty_like(J)
#        for w in range(len(self.weights)-1):
#            s = sum(self.weights[w]**2)
#            s += sum(self.weights[w+1]**2)
#            s *= 0.5*(self.Lambda/2)
#            print("sum",s)
#        J+=1    
        return J
        
    def costFunctionPrime(self, X, y):
        #Computer derivative with respect to W1, W2, W3, and W4
        self.yHat = self.forward(X)
        for i in range(len(self.z)-1, -1, -1):
            if len(self.z)>1:
                if i==len(self.z)-1:
                    self.deltas[i] = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z[i]))
                elif i==0:
                    self.deltas[i] = np.dot(self.deltas[i+1], self.weights[i+1].T)*self.sigmoidPrime(self.z[i])
                    self.grads[i] = np.dot(X.T, self.deltas[i])/X.shape[0] + self.Lambda*self.weights[i]
                    djs = self.grads
                    return djs
                else:
                    self.deltas[i] = np.dot(self.deltas[i+1], self.weights[i+1].T)*self.sigmoidPrime(self.z[i])
                self.grads[i] = np.dot(self.a[i-1].T, self.deltas[i])/X.shape[0] + self.Lambda*self.weights[i]
            else:
                self.deltas[i] = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z[i]))
                self.grads[i] = np.dot(X.T, self.deltas[i])/X.shape[0] + self.Lambda*self.weights[i]
                djs = self.grads
                return djs
            
        
    def getParams(self):
        #Get all Ws rolled into a vector:
        params = np.empty(0)
        for i in range(len(self.weights)):
            params = np.concatenate((params,self.weights[i].ravel()))
        return params
        
    def setParams(self, params):
        #Set all Ws using single parameter vector:
        start = 0
        end = -1
        for i in range(len(self.weights)):
            if not self.hiddenLayerSizes==[]:
                if i==0:
                    end = self.hiddenLayerSizes[i]*self.inputLayerSize
                    self.weights[i] = np.reshape(params[start:end], (self.inputLayerSize, self.hiddenLayerSizes[i]))
                    start = end
                elif i==len(self.weights)-1:
                    end += self.hiddenLayerSizes[i-1]*self.outputLayerSize
                    self.weights[i] = np.reshape(params[start:end], (self.hiddenLayerSizes[i-1], self.outputLayerSize))
                else:
                    end += self.hiddenLayerSizes[i-1]*self.hiddenLayerSizes[i]
                    self.weights[i] = np.reshape(params[start:end], (self.hiddenLayerSizes[i-1], self.hiddenLayerSizes[i]))
                    start = end
            else:
                end = self.outputLayerSize*self.inputLayerSize
                self.weights[i] = np.reshape(params[start:end], (self.inputLayerSize, self.outputLayerSize))
                start = end
        
    def computeGradients(self, X, y):
        djs = self.costFunctionPrime(X,y)
        p = []
        for e in djs:
            p = np.concatenate((p, e.ravel()))
        return p
        
class Trainer(object):
    def __init__(self,N):
        #Make local reference to neural network
        self.N = N
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X,y)
        grad = self.N.computeGradients(X,y)
        return cost,grad
        
    def callBackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X,self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))
    
    def train(self, trainX,trainY,testX,testY):
        #Make internal variable for callback function
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY
        #Make empty lists to store costs
        self.J = []
        self.testJ = []
        params0 = self.N.getParams()        
        options = {'maxiter':2000, 'disp':True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(trainX,trainY), options=options, callback=self.callBackF)
        self.N.setParams(_res.x)
        self.optimizationresults = _res
        
def computeNumericalGradient(N,X,y):
    paramsInitial = N.getParams()
    numgrad = np.empty_like(paramsInitial)
    perturb = np.zeros(paramsInitial.shape)        
    e = 0.000001
    for p in range(len(paramsInitial)):
        #Set perturbation vector
        perturb[p] = e
        N.setParams(paramsInitial+perturb)
        loss2 = N.costFunction(X,y)
        
        N.setParams(paramsInitial-perturb)
        loss1 = N.costFunction(X,y)
        
        #Compute Numerical Gradient
        numgrad[p] = (loss2-loss1)/2*e
        
        #Return value of perturb
        perturb[p] = 0
    #Return aprams to initial value
    N.setParams(paramsInitial)
    
    return numgrad

def parse():
    csv = open("D:/Documents/parameter-search/PredPrey/machine_learning/results/predprey_data.csv","r")
    
    lcount=0
    cmax = 0
    pmax = 0
    print("Checking files\n")
    for li in csv:
        print(lcount)
        sp = li.rstrip().split(",")
        params = np.zeros(8)
        fitnesses = np.zeros(3)
        for i in range(len(params)):
            params[i] = float(sp[i+1])
        fitnesses[0] = float(sp[10])
        fitnesses[1] = float(sp[11])
        fitnesses[2] = float(sp[12])
        if fitnesses[0]>cmax:
            cmax = fitnesses[0]
        if fitnesses[1]>pmax:
            pmax = fitnesses[1]
        if lcount==0:
            X = np.array([params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]],dtype=float)
            Y = np.array([fitnesses[0],fitnesses[2]],dtype=float)
            Y3 = np.array([fitnesses[0],fitnesses[1],fitnesses[2]],dtype=float)
        elif lcount==1:
            X = np.concatenate([[X],[params]])
            Y = np.concatenate([[Y],[[fitnesses[0],fitnesses[2]]]])
            Y3 = np.concatenate([[Y3],[fitnesses]])
        else:
            X = np.append(X,[params],axis=0)
            Y = np.append(Y,[[fitnesses[0],fitnesses[2]]],axis=0)
            Y3 = np.append(Y3,[fitnesses],axis=0)
        lcount+=1
    print("Dumping X, Y 2 fitnesses, and Y 3 Fitnesses\n")
    pickle.dump(X,open("X.p","wb"))
    pickle.dump(Y,open("Y2.p","wb"))
    pickle.dump(Y3,open("Y3.p","wb"))
    return X,Y

def parseFitness():
    csv = open("D:/Documents/parameter-search/PredPrey/machine_learning/results/predprey_data.csv","r")
    lcount=0
    print("Checking files\n")
    for li in csv:
        sp = li.rstrip().split(",")
        params = np.zeros(8)
        fitnesses = np.zeros(3)
        for i in range(len(params)):
            params[i] = float(sp[i+1])
        fitnesses[0] = float(sp[10])
        fitnesses[1] = float(sp[11])
        fitnesses[2] = float(sp[12])
        if lcount==0:
            f1 = np.array([fitnesses[0]],dtype=float)
            f2 = np.array([fitnesses[1]],dtype=float)
            f3 = np.array([fitnesses[2]],dtype=float)
#        elif lcount==1:
#            f1 = np.concatenate([f1,[fitnesses[0]]],axis=0)
#            f2 = np.concatenate([f2,[fitnesses[1]]],axis=0)
#            f3 = np.concatenate([f3,[fitnesses[2]]],axis=0)
        else:
            f1 = np.concatenate([f1,[fitnesses[0]]],axis=0)
            f2 = np.concatenate([f2,[fitnesses[1]]],axis=0)
            f3 = np.concatenate([f3,[fitnesses[2]]],axis=0)
        lcount+=1
    pickle.dump(f1,open("change.p","wb"))
    pickle.dump(f2,open("pop.p","wb"))
    pickle.dump(f3,open("nonz.p","wb"))

def main():
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print("seed",seed)
    #X,Y = parse()
    X = pickle.load(open("X.p","rb"))

    #Y = pickle.load(open("Y2.p","rb"))
    Y = pickle.load(open("Y3.p","rb"))
#    f1 = pickle.load(open("change.p","rb"))
#    f2 = pickle.load(open("pop.p","rb"))
#    f3 = pickle.load(open("nonz.p","rb"))
    #train = 0.95*len(X)
    #train = int(train)
    
    trainX,testX,trainY,testY = train_test_split(X,Y)
    
    
    #Y1 = Y1/cmax
    #Y2 = Y2/pmax
    #Y = np.concatenate([Y1,Y2],axis=1)
    #Y = np.concatenate([Y,Y3],axis=1)
    
#    trainX = X[0:train]
#    #trainY = Y[0:train]
#    trf1 = f1[0:train]
#    trf2 = f2[0:train]
#    trf3 = f3[0:train]
#    trf1 = trf1.reshape(-1,1)
#    trf2 = trf2.reshape(-1,1)
#    trf3 = trf3.reshape(-1,1)
    
    mmscalerX = preprocessing.MinMaxScaler(feature_range=(0,1))#MaxAbsScaler()#
    mmscalerX = mmscalerX.fit(trainX)
    print("X scale",mmscalerX)
    mmscalerY = preprocessing.MinMaxScaler(feature_range=(0,1))#MaxAbsScaler()#
    mmscalerY = mmscalerY.fit(trainY)
    print("Y scale", mmscalerY)
#    mmscalerf1 = preprocessing.MinMaxScaler(feature_range=(-1,1))#MaxAbsScaler()#
#    mmscalerf1 = mmscalerf1.fit(trf1)
#    print("f1 scale", mmscalerf1)
#    mmscalerf2 = preprocessing.MinMaxScaler(feature_range=(-1,1))#MaxAbsScaler()#
#    mmscalerf2 = mmscalerf2.fit(trf2)
#    print("f2 scale", mmscalerf2)
#    mmscalerf3 = preprocessing.MinMaxScaler(feature_range=(-1,1))#MaxAbsScaler()#
#    mmscalerf3 = mmscalerf3.fit(trf3)
#    print("f3 scale", mmscalerf3)
    pickle.dump(mmscalerX,open("scaleX.p","wb"))
    pickle.dump(mmscalerY,open("scaleY.p","wb"))
#    pickle.dump(mmscalerf1,open("scalef1.p","wb"))
#    pickle.dump(mmscalerf2,open("scalef2.p","wb"))
#    pickle.dump(mmscalerf3,open("scalef3.p","wb"))
    
    trainX = mmscalerX.transform(trainX)
    trainY = mmscalerY.transform(trainY)
#    trf1 = mmscalerf1.transform(trf1)
#    trf2 = mmscalerf2.transform(trf2)
#    trf3 = mmscalerf3.transform(trf3)
    
#    testX = X[train:len(X)]
#    testY = Y[train:len(Y)]
#    tef1 = f1[train:len(f1)]
#    tef2 = f2[train:len(f2)]
#    tef3 = f3[train:len(f3)]
#    tef1 = tef1.reshape(-1,1)
#    tef2 = tef2.reshape(-1,1)
#    tef3 = tef3.reshape(-1,1)
    
    testX = mmscalerX.transform(testX)
    testY = mmscalerY.transform(testY)
#    tef1 = mmscalerf1.transform(tef1)
#    tef2 = mmscalerf2.transform(tef2)
#    tef3 = mmscalerf3.transform(tef3)
    
    print("Initialising Neural networks\n")
#    NNC = Neural_Net(input_size=8, output_size=1, hidden_layers=0, hidden_layer_sizes=[], Lambda=0.0001)    
#    NNP = Neural_Net(input_size=8, output_size=1, hidden_layers=0, hidden_layer_sizes=[], Lambda=0.0001)    
#    NNN = Neural_Net(input_size=8, output_size=1, hidden_layers=0, hidden_layer_sizes=[], Lambda=0.0001)    
    NN = Neural_Net(input_size=8, output_size=3, hidden_layers=0, hidden_layer_sizes=[], Lambda=0.0001)    
    #print("Finding initial gradients and difference")
    #numgrad = computeNumericalGradient(NN,trainX,trainY)
    #print("step 2")
    #grad = NN.computeGradients(trainX,trainY)
    #print("step 3")
    #diff = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
    #print("Initial difference",diff)
    print("Training neural network\n")
    T = Trainer(NN)
    T.train(trainX,trainY,testX,testY)
    print("weights: \n",NN.weights)
    print("function: \n", T.optimizationresults.fun)
    pickle.dump(NN,open("NN.p","wb"))
#    T1 = Trainer(NNC)
#    T1.train(trainX,trf1,testX,tef1)
#    print("weights: \n",NNC.weights)
#    print("function: \n", T1.optimizationresults.fun)
#    pickle.dump(NNC,open("NNC.p","wb"))
#    T2 = Trainer(NNP)
#    T2.train(trainX,trf2,testX,tef2)
#    print("weights: \n",NNP.weights)
#    print("function: \n", T2.optimizationresults.fun)
#    pickle.dump(NNC,open("NNP.p","wb"))
#    T3 = Trainer(NNN)
#    T3.train(trainX,trf3,testX,tef3)
#    print("weights: \n",NNN.weights)
#    print("function: \n", T3.optimizationresults.fun)
#    pickle.dump(NNN,open("NNN.p","wb"))
    plt.figure()
    plt.figure(figsize=(16, 8))
    plt.plot(T.J, 'bo', label="J")
    plt.plot(T.testJ, label="testJ")
    plt.legend(frameon=True, fontsize=14, loc=1)
    plt.grid(1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig("D:/Documents/parameter-search/PredPrey/machine_learning/results/neural_network.png", bbox_inches='tight')

def test():
    trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
    trainY = np.array(([75], [82], [93], [70]), dtype=float)
    trainX = trainX/np.amax(trainX, axis=0)
    trainY = trainY/100
    
    #Testing Data:
    testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
    testY = np.array(([70], [89], [85], [75]), dtype=float)
    
    testX = testX/np.amax(testX, axis=0)
    testY = testY/100
    
    NN = Neural_Net(input_size=2, output_size=1, hidden_layers=0, hidden_layer_sizes=[], Lambda=0.0001)
    print("Finding intial gradients and difference")
    numgrad = computeNumericalGradient(NN,trainX,trainY)
    grad = NN.computeGradients(trainX,trainY)
    diff = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
    print("difference",diff)
    print("training NN")
    T = Trainer(NN)
    T.train(trainX,trainY,testX,testY)
    print("weights: \n",NN.weights)
    print("function: \n", T.optimizationresults.fun)
    pickle.dump(NN,open("NN.p","wb"))
    plt.figure()
    plt.figure(figsize=(16, 8))
    plt.plot(T.J, 'bo', label="J")
    plt.plot(T.testJ, label="testJ")
    plt.legend(frameon=True, fontsize=14, loc=1)
    plt.grid(1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    return

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

def testNN():
    NN = pickle.load(open("NN.p","rb"))
#    NNC = pickle.load(open("NNC.p","rb"))
#    NNP = pickle.load(open("NNP.p","rb"))
#    NNN = pickle.load(open("NNN.p","rb"))
    xscale = pickle.load(open("scaleX.p","rb"))
    yscale = pickle.load(open("scaleY.p","rb"))
#    f1s = pickle.load(open("scalef1.p","rb"))
#    f2s = pickle.load(open("scalef2.p","rb"))
#    f3s = pickle.load(open("scalef3.p","rb"))
    test_params=[[557,380,2200,0.131234,0.025769,25,40,38]]#,[2244,802,1876,0.132587,0.010532,119,19,46],[1391,2568,2927,0.159585,0.060718,31,18,31]]
    print("test",test_params)
    tp = xscale.transform(test_params)
    print("scaled test",tp)
#    prediction1 = NNC.forward(test_params)
#    prediction2 = NNP.forward(test_params)
#    prediction3 = NNN.forward(test_params)
    prediction = NN.forward(test_params)
    #print("test prediction for",mmscalerX.inverse_transform([testX[0]]))
    print("prediction",prediction)
#    print("prediction",prediction1,prediction2,prediction3)
    print("scaled prediction",yscale.inverse_transform(prediction))
    #print("scaled prediction",f1s.inverse_transform(prediction1),f2s.inverse_transform(prediction2),f3s.inverse_transform(prediction3))
    
    #print("scaled prediction",yscale.inverse_transform([prediction[2]]))
    print("real results:",[[56,87.238998,1],[57,171.455994,1],[54,375.036987,1]])
    return

def newscales():
    X = pickle.load(open("X.p","rb"))

    Y = pickle.load(open("Y3.p","rb"))
    train = 0.95*len(X)
    train = int(train)    
    trainX = X[0:train]
    trainY = Y[0:train]
    
    mmscalerX = preprocessing.MinMaxScaler(feature_range=(-1,1))
    mmscalerX = mmscalerX.fit(trainX)
    print("X scale",mmscalerX)
    mmscalerY = preprocessing.MinMaxScaler(feature_range=(-1,1))
    mmscalerY = mmscalerY.fit(trainY)
    print("Y scale", mmscalerY)
    pickle.dump(mmscalerX,open("scaleX.p","wb"))
    pickle.dump(mmscalerY,open("scaleY.p","wb"))
    return

main()
testNN()
#newscales()

#test()
#parse()
#parseFitness()
