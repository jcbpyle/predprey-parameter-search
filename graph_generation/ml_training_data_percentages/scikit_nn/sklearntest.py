# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:51:40 2017

@author: James
"""
import sys
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, confusion_matrix, classification_report
#import neural_network
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns
try:
    import cPickle as pickle
except:
    import pickle

sns.set_context(rc={"lines.linewidth": 3})
sns.set_style("whitegrid")

def main():
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print("seed",seed)
    X = pickle.load(open("X.p","rb"))
    c = pickle.load(open("change.p","rb"))
    p = pickle.load(open("pop.p","rb"))
    n = pickle.load(open("nonz.p","rb"))
    Y = []
    for i in range(len(c)):
        Y.append([c[i],p[i],n[i]])
        
    t = list(zip(X,Y))
    random.shuffle(t)
    te = t[:250000]
    X,Y = zip(*te)
    
    #Scaling
    mmscalerX = StandardScaler()
#    mmscalerX = preprocessing.MinMaxScaler()
#    mmscalerX = preprocessing.MaxAbsScaler()
#    mmscalerX = preprocessing.RobustScaler(quantile_range=(25, 75))
#    mmscalerX = preprocessing.QuantileTransformer(output_distribution='uniform')
#    mmscalerX = preprocessing.Normalizer()
    
    test_sizes = [0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.999]
#    test_sizes = [0.05, 0.1]
    
    single_score(X,Y,mmscalerX,test_sizes)
    #multi_score(X,Y,mmscalerX,test_sizes)
    #sklearn_regressor(X,Y,trainX,trainY,testX,testY,5)
    #sklearn_multiple(X,c,p,n,trainX,[try0,try1,try2],testX,[tey0,tey1,tey2],200) 
    
    return

def single_score(X,Y,scaler,test):
    scores = []
    for ts in test:
        trainX,testX,trainY,testY = train_test_split(X,Y, test_size=ts)
        scaler = scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)
        mlp = MLPRegressor(max_iter=2000, verbose=False, tol=0.000000001)
        mlp.fit(trainX,trainY)
        score = mlp.score(testX,testY)
        print("\n\n\n TEST Size:",ts)
        print("Score test: ",score)
        scores.append([ts,score])
    print(scores)
    pickle.dump(scores,open("singlenn_training_scores.p","wb"))
    sc = pickle.load(open("singlenn_training_scores.p","rb"))
    print(sc)
    return

def multi_score(X,Y,scaler,test):
    scores = []
    for ts in test:
        trainX,testX,trainY,testY = train_test_split(X,Y, test_size=ts)
        try0 = []
        try1 = []
        try2 = []
        try3 = []
        tey0 = []
        tey1 = []
        tey2 = []
        tey3 = []
        for j in range(len(trainY)):
            try0.append(trainY[j][0])
            try1.append(trainY[j][1])
            try2.append(trainY[j][2])
            try3.append([trainY[j][0],trainY[j][2]])
        for k in range(len(testY)):
            tey0.append(testY[k][0])
            tey1.append(testY[k][1])
            tey2.append(testY[k][2])
            tey3.append([testY[k][0],testY[k][2]])
        
        scaler = scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)
        scores.append(train_multiple(trainX,[try0,try1,try2],testX,[tey0,tey1,tey2],2000,ts))
    print(scores)
    pickle.dump(scores,open("training_scores.p","wb"))
    sc = pickle.load(open("training_scores.p","rb"))
    print(sc)
    return

def train_multiple(trainX,trainY,testX,testY,it,ts):
    mlp0 = MLPRegressor(max_iter=it, verbose=False, tol=0.000000001)
    mlp1 = MLPRegressor(max_iter=it, verbose=False, tol=0.000000001)
    mlp2 = MLPClassifier(max_iter=it, verbose=False, tol=0.000000001)
    mlp0.fit(trainX,trainY[0])
    mlp1.fit(trainX,trainY[1])
    mlp2.fit(trainX,trainY[2])
    score0 = mlp0.score(testX,testY[0])
    score1 = mlp1.score(testX,testY[1])
    score2 = mlp2.score(testX,testY[2])
    print("\n\n\n TEST Size:",ts)
    print("Score 0 test: ",score0)
    print("Score 1 test: ",score1)
    print("Score 2 test: ",score2)
    return [ts,score0,score1,score2]

def sklearn_regressor(X,Y,trainX,trainY,testX,testY,it):
    mlp = MLPRegressor(max_iter=it, verbose=False, tol=0.000000001)
    train_sizes, train_scores, valid_scores = learning_curve(mlp, X, Y, train_sizes=[0.95], cv=5)
#    print(train_sizes)
#    print(train_scores)
#    print(valid_scores)
    mlp.fit(trainX,trainY)
    predictions = mlp.predict(testX)
    print("MAE: ",mean_absolute_error(testY,predictions))
    print("MSE: ",mean_squared_error(testY,predictions))
    print("Explained variance: ",explained_variance_score(testY,predictions))
    print("Score all: ",mlp.score(X,Y))
    print("Score train: ",mlp.score(trainX,trainY))
    print("Score test: ",mlp.score(testX,testY))
    pickle.dump(mlp,open("mlp.p","wb"))
    
    plt.figure()
    plt.figure(figsize=(16, 8))
    plt.title("Testing SKlearn NN training", fontsize=24, color='black')
    plt.grid(1)
    plt.xlabel('Training examples', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.plot(train_scores[0], label="Training score")
    plt.plot(valid_scores[0], label="Validation score")
    plt.legend(frameon=True, fontsize=14, loc=1)
    plt.savefig("test_sknn.png", bbox_inches='tight')
    plt.savefig("test_sknn.pdf", bbox_inches='tight')
    return

def sklearn_multiple(X,c,p,n,trainX,trainY,testX,testY,it):
    mlp0 = MLPRegressor(max_iter=it, verbose=False, tol=0.000000001)
    train_sizes0, train_scores0, valid_scores0 = learning_curve(mlp0, X, c, train_sizes=[0.95], cv=50)
    print("Trained nn for fitness 1")
    plt.figure()
    plt.figure(figsize=(16, 8))
    plt.title("Training and Cross-Validation Score for Fitness 1", fontsize=24, color='black')
    plt.grid(1)
    plt.xlabel('Training examples', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.ylim(-1, 1)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.plot(train_scores0[0], label="Training score")
    plt.plot(valid_scores0[0], label="Validation score")
    plt.legend(frameon=True, fontsize=14, loc=3)
    plt.savefig("sknn_fitness1.png", bbox_inches='tight')
    plt.savefig("sknn_fitness1.pdf", bbox_inches='tight')
    sklm_aux(X,it,p,n)
    return

def sklm_aux(X,it,p,n):
    mlp1 = MLPRegressor(max_iter=it, verbose=False, tol=0.000000001)
    train_sizes1, train_scores1, valid_scores1 = learning_curve(mlp1, X, p, train_sizes=[0.95], cv=50)
    print("Trained nn for fitness 2")
    plt.figure()
    plt.figure(figsize=(16, 8))
    plt.title("Training and Cross-Validation Score for Fitness 2", fontsize=24, color='black')
    plt.grid(1)
    plt.xlabel('Training examples', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.ylim(-1, 1)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.plot(train_scores1[0], label="Training score")
    plt.plot(valid_scores1[0], label="Validation score")
    plt.legend(frameon=True, fontsize=14, loc=3)
    plt.savefig("sknn_fitness2.png", bbox_inches='tight')
    plt.savefig("sknn_fitness2.pdf", bbox_inches='tight')
    sklm_aux2(X,it,n)
    return

def sklm_aux2(X,it,n):
    mlp2 = MLPClassifier(max_iter=it, verbose=False, tol=0.000000001)
    train_sizes2, train_scores2, valid_scores2 = learning_curve(mlp2, X, n, train_sizes=[0.95], cv=50)
    print("Trained nn for fitness 3")
    plt.figure()
    plt.figure(figsize=(16, 8))
    plt.title("Training and Cross-Validation Score for Fitness 3", fontsize=24, color='black')
    plt.grid(1)
    plt.xlabel('Training examples', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.ylim(-1, 1)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.plot(train_scores2[0], label="Training score")
    plt.plot(valid_scores2[0], label="Validation score")
    plt.legend(frameon=True, fontsize=14, loc=3)
    plt.savefig("sknn_fitness3.png", bbox_inches='tight')
    plt.savefig("sknn_fitness3.pdf", bbox_inches='tight')
#    predictions0 = mlp0.predict(testX)
#    print(confusion_matrix(testY[0],predictions0))
#    print(classification_report(testY[0],predictions0))
#    predictions1 = mlp1.predict(testX)
#    print(confusion_matrix(testY[1],predictions1))
#    print(classification_report(testY[1],predictions1))
#    predictions2 = mlp2.predict(testX)
#    print(confusion_matrix(testY[2],predictions2))
#    print(classification_report(testY[2],predictions2))
    return

def generate_graphs():
    single = pickle.load(open("singlenn_training_scores.p","rb"))
    print(single)
    multi = pickle.load(open("training_scores.p","rb"))
    print(multi)
    plt.figure(figsize=(12, 6))
    plt.title("Training scores for neural networks estimating simulation output(s)", fontsize=24, color='black')
    plt.xlabel("Test data size", fontsize=18)
    plt.ylabel("Neural network score", fontsize=16)
    plt.yticks(fontsize=12)
    #plt.xlim(0, 10)
    #plt.ylim(0, 10000)
    plt.xticks(fontsize=12)
    plt.plot([s[0] for s in single], [s[1] for s in single], 'b-', label="All 3 fitnesses")
    plt.plot([m[0] for m in multi], [m[1] for m in multi], 'r-', label="Fitness 1")
    plt.plot([m[0] for m in multi], [m[2] for m in multi], 'g-', label="Fitness 2")
    plt.plot([m[0] for m in multi], [m[3] for m in multi], 'c-', label="Fitness 3")
    plt.legend(frameon=True, fontsize=18, loc=3)
    plt.tight_layout()
    plt.savefig(os.getcwd()+"/neural_network_scores.png", bbox_inches='tight')
    plt.savefig(os.getcwd()+"/neural_network_scores.pdf", bbox_inches='tight')
    return

generate_graphs()
#main()