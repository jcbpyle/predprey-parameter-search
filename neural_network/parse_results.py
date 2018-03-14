# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:49:49 2017

@author: James
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
RESULTS_PATH = os.getcwd()+"/../mavericks_results/ML/"
SAVE_DIRECTORY = os.getcwd()+"/results"
open(SAVE_DIRECTORY+"/predprey_data.csv", "a").close()

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111,projection='3d')
def parse():
    directories = [x[0] for x in os.walk(RESULTS_PATH)]
    for folder in directories:
        print(folder)
        if os.path.exists(folder+"/simulation_results.csv"):
            csv = open(folder+"/simulation_results.csv")
            r = csv.readlines()
            csv.close()
            p = open(SAVE_DIRECTORY+"/predprey_data.csv", "a")
            p.writelines(r)
            p.close()
    return

def parse2():
    df = pd.DataFrame(data=None)
    csv = open(SAVE_DIRECTORY+"/predprey_data.csv", "r")
    for line in csv:
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
        
        
        d1 = {'results':(params, fitnesses)}
        ndf = pd.DataFrame(data=d1)
        df = pd.concat([df,ndf], axis=0)
        string = ""
        for i in params:
            string += str(i)+","
        ax.scatter(fitnesses[0],fitnesses[1],fitnesses[2], label=string)
#parse()
parse2()
ax.set_xlabel("Changeover")
ax.set_ylabel("Population Difference")
ax.set_zlabel("Non Zero")
fig.savefig(SAVE_DIRECTORY+"/parameter_space_fitness.png", bbox_inches='tight')
#plt.plot(thing['results'][0],thing['results'][1])
#print(thing)