# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:41:19 2017

@author: James
"""

##############################################################################
#Batch job script for generating data for parameter sets simulating the Predator/Prey model
import random
import os
import sys
import threading
import pycuda.driver as cuda
import pycuda.autoinit
#from pycuda.compiler import SourceModule
import argparse
#import pandas as pd
import matplotlib.pyplot as plt

global toolbox
global hof
global s1,s2,s3


parser = argparse.ArgumentParser(description='Python implementation of parameter search using genetic algorithms for predprey')
parser.add_argument('numsets', metavar='parameter_sets', type=int, nargs='+',
                   help='The number of parameter sets to generate simulation data about')
parser.add_argument('generations', metavar='gens', type=int, nargs='+',
                   help='The number of generations to run each parameter set for')
#parser.add_argument('stats', metavar='statruns', type=int, nargs='+',
                   #help='The number of runs to perform each parameter set for')
parser.add_argument('maxpopsize', metavar='maxpop', type=int, nargs='+',
                   help='The maximum size of agent populations in new parameter sets')

args = parser.parse_args()

#Editable run specification parameters
GENERATIONS = args.generations[0]
PARAMS = int(args.numsets[0])

#GPU availability
GPUS_AVAILABLE = cuda.Device(0).count()
#STAT_RUNS = GPUS_AVAILABLE*5

#Set random seed
seed = random.randrange(sys.maxsize)
random.seed(seed)

#Set file save paths
SAVEPATH = "results/"
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)
PATH_TO_CURR = os.getcwd()+"/"

#Initial parameter values for predprey model
PREY_RANGE_MIN = 0
PREY_RANGE_MAX = args.maxpopsize[0]
PRED_RANGE_MIN = 0
PRED_RANGE_MAX = args.maxpopsize[0]
GRASS_RANGE_MIN = 0
GRASS_RANGE_MAX = args.maxpopsize[0]
PREY_REPRODUCTION_RANGE_MIN = 0.0
PREY_REPRODUCTION_RANGE_MAX = 0.25
PRED_REPRODUCTION_RANGE_MIN = 0.0
PRED_REPRODUCTION_RANGE_MAX = 0.25
PREY_ENERGY_GAIN_RANGE_MIN = 0
PREY_ENERGY_GAIN_RANGE_MAX = 200
PRED_ENERGY_GAIN_RANGE_MIN = 0
PRED_ENERGY_GAIN_RANGE_MAX = 200
GRASS_REGROW_RANGE_MIN = 0
GRASS_REGROW_RANGE_MAX = 200

def main():
    for i in range(PARAMS):
        threads = []
        for g in range(GPUS_AVAILABLE):
            thread = dataThread(g)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
    return

def threadFunction(d):
    x = generateParameterSet()
    if not os.path.exists(SAVEPATH+str(d)):
        os.makedirs(SAVEPATH+str(d))
    if not os.path.exists(SAVEPATH+str(d)+"/0.xml"):
        open(SAVEPATH+str(d)+"/0.xml", "w").close()
    
    if os.name=='nt':
        genComm = "xmlGenEx3.exe "+SAVEPATH+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])
        command = "PreyPredator.exe "+SAVEPATH+str(d)+"/0.xml "+str(GENERATIONS)+" "+str(d)
    else:
        genComm = "./xmlGenEx3 "+SAVEPATH+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])        
        command = "./PreyPredator_console "+SAVEPATH+str(d)+"/0.xml "+str(GENERATIONS)+" "+str(d)
    os.system(genComm)
    os.system(command)
        
class dataThread(threading.Thread):
    def __init__(self, device):
        threading.Thread.__init__(self)
        self.device = device
    def run(self):
        threadFunction(self.device)
        
def generateParameterSet():
    new = [0]*8
    new[0] = int(random.uniform(PREY_RANGE_MIN, PREY_RANGE_MAX))
    new[1] = int(random.uniform(PRED_RANGE_MIN, PRED_RANGE_MAX))
    new[2] = int(random.uniform(GRASS_RANGE_MIN, GRASS_RANGE_MAX))
    new[3] = random.uniform(PREY_REPRODUCTION_RANGE_MIN, PREY_REPRODUCTION_RANGE_MAX)
    new[4] = random.uniform(PRED_REPRODUCTION_RANGE_MIN, PRED_REPRODUCTION_RANGE_MAX)
    new[5] = int(random.uniform(PRED_ENERGY_GAIN_RANGE_MIN, PRED_ENERGY_GAIN_RANGE_MAX))
    new[6] = int(random.uniform(PREY_ENERGY_GAIN_RANGE_MIN, PREY_ENERGY_GAIN_RANGE_MAX))
    new[7] = int(random.uniform(GRASS_REGROW_RANGE_MIN, GRASS_REGROW_RANGE_MAX))
    return new

main()

#plt.plot(logbook.select("c-avg"), 'green', label='Population Changeover')
##plt.plot(logbook.select("p-avg"), 'blue', label='Absolute Population Difference')
#plt.plot(logbook.select("n-avg"), 'red', label='Non Zero Population')
##Make and show plotted data
#plt.legend(frameon=True, fontsize=18, loc=1)
#plt.title("Fitness over time", fontsize=24, color='black')
#plt.xlabel("GA Generations", fontsize=18)
#plt.ylabel("Fitness", fontsize=18)
#plt.yticks(fontsize=10)
#plt.xlim(0, int((maxevals-MU)/LAMBDA)+1)
##plt.ylim(0, 10000)
#plt.xticks(fontsize=10)
##plt.savefig(PATH+"/ga_results/graphs/"+title+".svg")
##Save image
#plt.tight_layout()
#plt.savefig(SAVEPATH+"graphs/test.png", bbox_inches='tight')
#plt.show() 