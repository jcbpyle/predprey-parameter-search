# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:21:32 2018

@author: James Pyle
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Set seaborn graphics
sns.set_context(rc={"lines.linewidth": 3})
sns.set_style("whitegrid")

#Parser arguments
parser = argparse.ArgumentParser(description='Python implementation of predator prey model data generation')
parser.add_argument('individual', metavar='individual', type=str, nargs='+',
                   help='The parameter set to generate data for. A string of the form "prey,pred,..."')
parser.add_argument('gens', metavar='sim_it', type=int, nargs='+',
                   help='The number of iterations to run the simulation of the parameter set for')
parser.add_argument('stats', metavar='stats', type=int, nargs='+',
                   help='The number of times to run the parameter set')

args = parser.parse_args()
ind = args.individual[0].split(",")
gens = args.gens[0]
stats = args.stats[0]
PATH = os.getcwd()
DIRECTORY = "/generated_graphs"

def generateData():
    #Create path for results
    if not os.path.exists(PATH+DIRECTORY):
        os.makedirs(PATH+DIRECTORY)
    #Set plot
    plt.figure(figsize=(10,5))   
    #Initialise parameter set
    pry = ind[0]
    prd = ind[1]
    g = ind[2]
    pyr = ind[3]
    pdr = ind[4]
    pye = ind[5]
    pde = ind[6]
    gr = ind[7]
    #Set figure information
    title = "prey"+str(pry)+"_pred"+str(prd)+"_grass"+str(g)
    d = {'prey':[], 'pred':[], 'grass':[]}
    df = pd.DataFrame(data=d)
    #Perform a number fo runs per parameter set
    for j in range(stats):
        genComm = "xmlGenEx3.exe "+PATH+DIRECTORY+"/"+str(j)+".xml "+str(pry)+" "+str(prd)+" "+str(g)+" "+str(pyr)+" "+str(pdr)+" "+str(pye)+" "+str(pde)+" "+str(gr)
        os.system(genComm)
    for k in range(stats):
        command = "PreyPredator.exe "+PATH+DIRECTORY+"/"+str(k)+".xml "+str(gens)
        os.system(command)
        #Save results for this run
        prey = np.zeros(gens)
        pred = np.zeros(gens)
        grass  = np.zeros(gens)
        lcount = 0
        csv = open(PATH+DIRECTORY+"/PreyPred_Count.csv","r")
        #csv = open(PATH+DIRECTORY+"/simulation_results.csv","r")
        for line in csv:
            if lcount<gens:
                sp = line.split(",")
                prey[lcount]=sp[1]
                pred[lcount]=sp[3]
                grass[lcount]=sp[5]
            lcount+=1
        d = {'prey':prey, 'pred':pred, 'grass':grass}
        ndf = pd.DataFrame(data=d)
        #Plot data and legend if first run
        if k==0:
            plt.plot(ndf['grass'], 'green', label='grass')
            plt.plot(ndf['prey'], 'blue', label='prey')
            plt.plot(ndf['pred'], 'red', label='pred')
        else:
            plt.plot(ndf['grass'], 'green', label='_nolegend_')
            plt.plot(ndf['prey'], 'blue', label='_nolegend_')
            plt.plot(ndf['pred'], 'red', label='_nolegend_')
        df = pd.concat([df, ndf], axis=1)
        csv.close()
    #Make and show plotted data
    plt.legend(frameon=True, fontsize=18, loc=1)
    plt.title(title, fontsize=24, color='black')
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Population Size", fontsize=18)
    plt.yticks(fontsize=14)
    plt.xlim(0, 1000)
    plt.ylim(0, 6500)
    plt.xticks(fontsize=14)
    #Save image
    plt.tight_layout()
    plt.savefig(PATH+DIRECTORY+"/"+title+".png", bbox_inches='tight')
    plt.savefig(PATH+DIRECTORY+"/"+title+".pdf", bbox_inches='tight')
    plt.show() 
    #Save all results from dataframe
    savefile = open(PATH+DIRECTORY+"/"+title+".txt", "a")
    savefile.write(df.to_string())
    savefile.close()
    return
    
generateData()