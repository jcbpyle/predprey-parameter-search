# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:14:15 2017

@author: James
"""

import numpy as np
import random
import sys
import os
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
try:
    import cPickle as pickle
except:
    import pickle

sns.set_context(rc={"lines.linewidth": 1})
sns.set_style("whitegrid")

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
    def __init__(self,N, it):
        #Make local reference to neural network
        self.N = N
        self.it = it
        
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
        options = {'maxiter':self.it, 'disp':True}
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
    csv = open(os.getcwd()+"/../predprey_data.csv","r")
    
    lcount=0
    cmax = 0
    pmax = 0
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
    csv = open(os.getcwd()+"/../predprey_data.csv","r")
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

def gen_params():
    new = [0]*8
    new[0] = int(random.uniform(0, 5000))
    new[1] = int(random.uniform(0, 5000))
    new[2] = int(random.uniform(0, 5000))
    new[3] = random.uniform(0, 0.25)
    new[4] = random.uniform(0, 0.25)
    new[5] = int(random.uniform(0, 250))
    new[6] = int(random.uniform(0, 250))
    new[7] = int(random.uniform(0, 250))
    return new

def train_NN(trainX,trainY,testX,testY,output,iterations,title,save):
    if (len(trainY[0])!=output):
        print("provided data doesn't match output neurons")
    else:
        print("Initialising neural network "+save+"\n")   
        NN = Neural_Net(input_size=8, output_size=output, hidden_layers=0, hidden_layer_sizes=[], Lambda=0.0001)
        print("Training neural network "+save+"\n")
        T = Trainer(NN,iterations)
        T.train(trainX,trainY,testX,testY)
        print("weights: \n",NN.weights)
        print("function: \n", T.optimizationresults.fun)
        pickle.dump(NN,open(save+".p","wb"))
        pickle.dump(T,open(save+"_trainer.p","wb"))
        
        plt.figure()
        plt.figure(figsize=(16, 8))
        plt.title(title, fontsize=24, color='black')
        plt.grid(1)
        plt.xlabel('Training iteration', fontsize=18)
        plt.ylabel('Cost', fontsize=18)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.tight_layout()
        plt.plot(T.J, 'bo', label="X cost")
        plt.plot(T.testJ, label="test_X cost")
        plt.legend(frameon=True, fontsize=14, loc=1)
        plt.savefig(save+".png", bbox_inches='tight')
        plt.savefig(save+".pdf", bbox_inches='tight')
    return

def main():
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print("seed",seed)
    X = pickle.load(open("X.p","rb"))
#    X = pickle.load(open("X.p","rb"))
#    Y = pickle.load(open("Y2.p","rb"))
#    Y = pickle.load(open("Y3.p","rb"))
    c = pickle.load(open("change.p","rb"))
    p = pickle.load(open("pop.p","rb"))
    n = pickle.load(open("nonz.p","rb"))
    Y = []
    for i in range(len(c)):
        Y.append([c[i],p[i],n[i]])
#    X, Y = zip(*random.sample(list(zip(X,Y)), 250000))
    
    trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.05)
    try0 = []
    try1 = []
    try2 = []
    try3 = []
    tey0 = []
    tey1 = []
    tey2 = []
    tey3 = []
    for j in range(len(trainY)):
        try0.append([trainY[j][0]])
        try1.append([trainY[j][1]])
        try2.append([trainY[j][2]])
        try3.append([trainY[j][0],trainY[j][2]])
    for k in range(len(testY)):
        tey0.append([testY[k][0]])
        tey1.append([testY[k][1]])
        tey2.append([testY[k][2]])
        tey3.append([testY[k][0],testY[k][2]])
        
    #Scaling
    mmscalerX = preprocessing.StandardScaler()
#    mmscalerX = preprocessing.MinMaxScaler()
#    mmscalerX = preprocessing.MaxAbsScaler()
#    mmscalerX = preprocessing.RobustScaler(quantile_range=(25, 75))
#    mmscalerX = preprocessing.QuantileTransformer(output_distribution='uniform')
#    mmscalerX = preprocessing.Normalizer()
    mmscalerX = mmscalerX.fit(trainX)
    trainX = mmscalerX.transform(trainX)
    testX = mmscalerX.transform(testX)
    
    train_NN(trainX,trainY,testX,testY,3,2000,"Training error for neural network to predict output of pred-prey simulation","neural_network")
    train_NN(trainX,try0,testX,tey0,1,2000,"Training error for neural network to predict fitness 1","neural_network0")
    train_NN(trainX,try1,testX,tey1,1,2000,"Training error for neural network to predict fitness 2","neural_network1")
    train_NN(trainX,try2,testX,tey2,1,2000,"Training error for neural network to predict fitness 3","neural_network2")
    train_NN(trainX,try3,testX,tey3,2,2000,"Training error for neural network to predict fitnesses 1 and 3","neural_network_only2")
    return

#main()