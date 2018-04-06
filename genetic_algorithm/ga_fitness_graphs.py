# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:15:54 2018

@author: James
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = os.getcwd()+"/mavericks_results/"

sns.set_context(rc={"lines.linewidth": 2})
sns.set_style("whitegrid")

def fitness_graph(g):
    gcount = 0
    f = open(RESULTS_PATH+"20+1_ga_generations.csv","r")
    f0 = []
    f1 = []
    f2 = []
    for l in f:
        if gcount<g:
            sp = l.split(",")
            if sp[0]=="Generation":
                gcount+=1
            else:
                f0.append([gcount,float(sp[11])])
                f1.append([gcount,float(sp[12])])
                f2.append([gcount,float(sp[13])])
    plt_fit([f[0] for f in f0], [f[1] for f in f0],"Fitness 1 over GA generations","Fitness_1")
    plt_fit([f[0] for f in f1], [f[1] for f in f1],"Fitness 2 over GA generations","Fitness_2")
    plt_fit([f[0] for f in f2], [f[1] for f in f2],"Fitness 3 over GA generations","Fitness_3")
    return

def plt_fit(x,y,title,save):
    plt.figure(figsize=(10, 4))
    plt.title(title, fontsize=24, color='black')
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Fitness", fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.plot(x,y, 'b-')
    plt.legend(frameon=True, fontsize=18, loc=2)
    plt.tight_layout()
    plt.savefig(os.getcwd()+"/"+save+".png", bbox_inches='tight')
    plt.savefig(os.getcwd()+"/"+save+".pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

fitness_graph(5000)