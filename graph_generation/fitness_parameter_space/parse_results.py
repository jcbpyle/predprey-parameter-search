# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:49:49 2017

@author: James
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
    import cPickle as pickle
except:
    import pickle
import seaborn as sns

sns.set_context(rc={"lines.linewidth": 1})
sns.set_style("whitegrid")

def graph_random_data_good():
    gp = pickle.load(open("parameter_space_fitness_good.p","rb"))
    plt.figure()
    plt.figure(figsize=(16, 8))
    X = []
    Y = []
    c = []
    for i in gp:
        X.append(gp[i][1][0])
        Y.append(gp[i][1][1])
        c.append(gp[i][2])
    m = cm.ScalarMappable(cmap=cm.Greens)
    m.set_array(c)
    plt.scatter(X,Y, c=c, vmin=0, vmax=20, s=35, cmap=m)
    cb = plt.colorbar(m)
    cb.set_label('Worse to better solutions', rotation=270, fontsize=16, labelpad=15)
    cb.ax.tick_params(labelsize=12)
    plt.title("Fitness range of parameter sets with >0 predators and prey at simulation termination, Fitness 3==1", fontsize=24, color='black')
    plt.grid(1)
    plt.xlabel('Fitness 1', fontsize=18)
    plt.ylabel('Fitness 2 (1,000s)', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.legend(frameon=True, fontsize=14, loc=1)
    plt.savefig("parameter_space_fitness_2d_good.png", bbox_inches='tight')
    plt.savefig("parameter_space_fitness_2d_good.pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

def graph_random_data_bad():
    bp = pickle.load(open("parameter_space_fitness_bad.p","rb"))
    plt.figure()
    plt.figure(figsize=(16, 8))
    X = []
    Y = []
    c = []
    for i in bp:
        X.append(bp[i][1][0])
        Y.append(bp[i][1][1])
        c.append(bp[i][2])
    m = cm.ScalarMappable(cmap=cm.Reds)
    m.set_array(c)
    plt.scatter(X,Y, c=c, vmin=0, vmax=20, s=35, cmap=m)
    cb = plt.colorbar(m)
    cb.set_label('Worse to better solutions', rotation=270, fontsize=16, labelpad=15)
    cb.ax.tick_params(labelsize=12)
    plt.title("Fitness of parameter sets with 0 predators or prey at simulation termination, Fitness 3==0", fontsize=24, color='black')
    plt.grid(1)
    plt.xlabel('Fitness 1', fontsize=18)
    plt.ylabel('Fitness 2 (1,000s)', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.legend(frameon=True, fontsize=14, loc=1)
    plt.savefig("parameter_space_fitness_2d_bad.png", bbox_inches='tight')
    plt.savefig("parameter_space_fitness_2d_bad.pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

def parse_random_data_2d():
    gp = pd.DataFrame(data=None)
    bp = pd.DataFrame(data=None)
    csv = open(os.getcwd()+"/../predprey_data.csv", "r")
    high_low = [[99999,0],[99999,0]]
    l = 0
    for line in csv:
        if l%5000==0:
            print("parse random: ",l)
        l+=1
        sp = line.rstrip().split(",")
        params = [0 for x in range(8)]
        fitnesses = [0 for x in range(3)]
        
        if sp[10] == "err":
            print(sp)
            continue
        for i in range(len(params)):
            if i==3 or i==4:
                params[i] = float(sp[i+1])
            else:
                params[i] = int(sp[i+1])
        for j in range(len(fitnesses)):
            fitnesses[j] = float(sp[i+j+3])
        fitnesses[1] /= 1000
        if fitnesses[0]<high_low[0][0]:
            high_low[0][0] = fitnesses[0]
        if fitnesses[0]>high_low[0][1]:
            high_low[0][1] = fitnesses[0]
        if fitnesses[1]<high_low[1][0]:
            high_low[1][0] = fitnesses[1]
        if fitnesses[1]>high_low[1][1]:
            high_low[1][1] = fitnesses[1]
        d1 = {'results':(params,fitnesses,None)}
        ndf = pd.DataFrame(data=d1)
        if fitnesses[2]==0:
            bp = pd.concat([bp,ndf], axis=1, ignore_index=True)
        else:
            gp = pd.concat([gp,ndf], axis=1, ignore_index=True)
    print("done parsing random data samples")
    change = high_low[0][1]-high_low[0][0]
    pod = high_low[1][1]-high_low[1][0]
    print("pickling random data")
    l = 0
    for i in bp:
        f = bp[i][1]
        if l%5000==0:
            print("pickling random: ",l)
        l+=1
        colour = (0.6,0,0)
        c0 = (f[0]-high_low[0][0])/change
        c1 = (f[1]-high_low[1][0])/pod
        col = (0.5+c0*(0.5))-(0.1+c1*(0.4))
        colour=(col,0,0)
        bp[i][2] = colour
    for j in gp:
        f = gp[j][1]
        colour = (0.6,0,0)
        c0 = (f[0]-high_low[0][0])/change
        c1 = (f[1]-high_low[1][0])/pod
        col = (0.5+c0*(0.5))-(0.1+c1*(0.4))
        colour=(0,col,0)
        gp[j][2] = colour
    pickle.dump(gp,open("parameter_space_fitness_good.p","wb"))
    pickle.dump(bp,open("parameter_space_fitness_bad.p","wb"))
    return

def main():
    #parse_random_data_2d()
    graph_random_data_bad()
    graph_random_data_good()
    return

main()