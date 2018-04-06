# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:47:43 2017

@author: James
"""

##############################################################################
#Batch job script for GA paramter searching Predator/Prey

from deap import base
from deap import tools
from deap import creator

import random
import numpy as np
import os
import sys
import threading
import pycuda.driver as cuda
import pycuda.autoinit
#from pycuda.compiler import SourceModule
import argparse
import datetime
import matplotlib.pyplot as plt

global toolbox
global hof
global s1,s2,s3


parser = argparse.ArgumentParser(description='Python implementation of parameter search using genetic algorithms for predprey')
parser.add_argument('pops', metavar='MU', type=int, nargs='+',
                   help='The number of parameter sets(chromosomes) to initialise')
parser.add_argument('lamb', metavar='LAMBDA', type=int, nargs='+',
                   help='The number of child parameter sets(chromosomes) to create each iteration')
parser.add_argument('GAevals', metavar='GAevals', type=int, nargs='+',
                   help='The number of GA fitness function evaluations to perform')
parser.add_argument('gens', metavar='gens', type=int, nargs='+',
                   help='The number of generations to run each parameter set for')
parser.add_argument('stats', metavar='statruns', type=int, nargs='+',
                   help='The number of runs to perform each parameter set for')
parser.add_argument('maxpopsize', metavar='maxpop', type=int, nargs='+',
                   help='The maximum size of agent populations in new parameter sets')

args = parser.parse_args()

#Editable run specification parameters
NUM_POPULATIONS = args.pops[0]
GENERATIONS = args.gens[0]

GPUS_AVAILABLE = cuda.Device(0).count()
STAT_RUNS = args.stats[0]

#Editable parameters for simulation variables and GA mutation
MAX_POP = args.maxpopsize[0]
MAX_ALTERATION = 0.25
NEW_OFFSPRING = 0.5

#Set random seed
seed = 2041057997383656593#random.randrange(sys.maxsize)
random.seed(seed)

#GA Parameters
MU = NUM_POPULATIONS
#if len(args.lamb>1):
#    LAMBDA = (args.lamb[0],args.lamb[1])
#    GATYPE = MU+"+"+LAMBDA[0]+"-"+LAMBDA[1]
#else:
LAMBDA = args.lamb[0]
GATYPE = str(MU)+"+"+str(LAMBDA)
crossover = True#args.crossover[0]
maxevals = args.GAevals[0]

SAVEPATH = "ga_results/"
PATH_TO_CURR = os.getcwd()+"/"

if not os.path.exists(PATH_TO_CURR+SAVEPATH+GATYPE):
        os.makedirs(PATH_TO_CURR+SAVEPATH+GATYPE)
if not os.path.exists(PATH_TO_CURR+SAVEPATH+"results_archive/"+GATYPE):
    os.makedirs(PATH_TO_CURR+SAVEPATH+"results_archive/"+GATYPE)
SAVEPATH += GATYPE+"/"

#Initial parameter values for predprey model
PREY_RANGE_MIN = 0
PREY_RANGE_MAX = MAX_POP
PRED_RANGE_MIN = 0
PRED_RANGE_MAX = MAX_POP
GRASS_RANGE_MIN = 0
GRASS_RANGE_MAX = MAX_POP
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
curr_pop = 0
changeover = 0.0
popratio = 0.0
nonzero = 0.0

#Example base individual population (one chromosome for GA purposes)
#In the form of:
#num_prey, num_pred, num_grass, prey_reproduction, pred_reproduction, pred_energy, 
#   prey_energy, grass_regrow, run_number
#base_indiv = [386, 1745, 2048, 0.13018519971742398, 0.09222382062602291, 156, 10, 69, 0]

#--------------------UTILITY Functions-------------------------#

def log(logbook, population, gen, nevals):
    record1 = s1.compile(population) if s1 else {}
    record2 = s2.compile(population) if s2 else {}
    record3 = s3.compile(population) if s3 else {}
    logbook.record(gen=gen, nevals=nevals, **record1, **record2, **record3)
    if hof is not None:
        hof.update(population)
  
def fitnessValue(individual):
    return individual.fitness.values[0]

def bestFitness(population):
    return tools.selBest(population, 1)[0]

def favourOffspring(parents, offspring, MU):
    choice = (list(zip(parents, [0]*len(parents))) +
              list(zip(offspring, [1]*len(offspring))))
    choice.sort(key=lambda x: (fitnessValue(x[0]), x[1]), reverse=True)
    return [x[0] for x in choice[:MU]]

def selectParents(toolbox, individuals, k):
    parents = [random.choice(individuals) for i in range(k)]
    return [toolbox.clone(ind) for ind in parents]





#---------------------------THREAD----------------------------#
class ppThread(threading.Thread):
    def __init__(self, device, individual, runs):
        threading.Thread.__init__(self)
        self.device = device
        self.individual = individual
        self.runs = runs
    def run(self):
        evalAuxillary(self.device, self.individual, self.runs)




#----------------PREDPREY GA Functions--------------------------#
#Evaluate an individual (parameter set) x in the population
def evalPP(x):
    #Measure changeover of pred and prey population sizes, the ratio of prey to
    #   pred, and if both populations are non zero at the end of the run
    global changeover
    global popratio
    global nonzero
    changeover = [0.0]*GPUS_AVAILABLE
    popratio = [0.0]*GPUS_AVAILABLE
    nonzero = [0.0]*GPUS_AVAILABLE
    threads = []
    for g in range(GPUS_AVAILABLE):
        thread = ppThread(g, x, int(STAT_RUNS/GPUS_AVAILABLE))
        thread.start()
        threads.append(thread)
    for t in threads:
        t.join()
    changeover = sum(changeover)
    popratio = sum(popratio)
    nonzero = sum(nonzero)
    #Average over the number of runs performed
    changeover = changeover/STAT_RUNS
    popratio = popratio/STAT_RUNS
    nonzero = nonzero/STAT_RUNS
    return changeover,popratio,nonzero,

def evalAuxillary(d, x, runs):
    global changeover
    global popratio
    global nonzero
    if not os.path.exists(SAVEPATH+str(d)):
        os.makedirs(SAVEPATH+str(d))
    if not os.path.exists(SAVEPATH+str(d)+"/0.xml"):
        open(SAVEPATH+str(d)+"/0.xml", "w").close()
    if not os.path.exists(SAVEPATH+str(d)+"/save.csv"):
        open(SAVEPATH+str(d)+"/save.csv", "w").close()
    for r in range(runs):
        open(SAVEPATH+str(d)+"/simulation_results.csv","w").close()
        if os.name=='nt':
            genComm = PATH_TO_CURR+"../xmlGenEx3.exe "+SAVEPATH+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])
            command = PATH_TO_CURR+"../PreyPredator.exe "+SAVEPATH+str(d)+"/0.xml "+str(GENERATIONS)+" "+str(d)
        else:
            genComm = "../xmlGenEx3 "+SAVEPATH+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])        
            command = "../PreyPredator_console "+SAVEPATH+str(d)+"/0.xml "+str(GENERATIONS)+" "+str(d)
        os.system(genComm)
        os.system(command)
        
        #csv file data
        csv = open(SAVEPATH+str(d)+"/simulation_results.csv","r")
        li =  csv.readline()
        s = li.split(",")
        changeover[d] = int(s[10])
        popratio[d] = float(s[11])
        nonzero[d] = int(s[12])
        csv.close()
        save = open(SAVEPATH+str(d)+"/save.csv","a")
        save.write(li)
        save.close()

#Create a new parameter set
def initIndividual(container):
    global curr_pop
    new = [0]*9
    new[0] = int(random.uniform(PREY_RANGE_MIN, PREY_RANGE_MAX))
    new[1] = int(random.uniform(PRED_RANGE_MIN, PRED_RANGE_MAX))
    new[2] = int(random.uniform(GRASS_RANGE_MIN, GRASS_RANGE_MAX))
    new[3] = round(random.uniform(PREY_REPRODUCTION_RANGE_MIN, PREY_REPRODUCTION_RANGE_MAX),6)
    new[4] = round(random.uniform(PRED_REPRODUCTION_RANGE_MIN, PRED_REPRODUCTION_RANGE_MAX),6)
    new[5] = int(random.uniform(PRED_ENERGY_GAIN_RANGE_MIN, PRED_ENERGY_GAIN_RANGE_MAX))
    new[6] = int(random.uniform(PREY_ENERGY_GAIN_RANGE_MIN, PREY_ENERGY_GAIN_RANGE_MAX))
    new[7] = int(random.uniform(GRASS_REGROW_RANGE_MIN, GRASS_REGROW_RANGE_MAX))
    new[8] = curr_pop
    curr_pop += 1
    return container(new)

#Initialise the population for deap
def initPopulation(container, ind_init):
    return container(ind_init(i) for i in range(NUM_POPULATIONS))

#Mutate a parameter set, only reproduction rates and energy gain eligible for mutation
def mutate(toolbox, individual):
    changes = np.random.choice([1,2,3,4], p=[0.6,0.2,0.1,0.1])
    ch = random.sample([3,4,5,6], changes)
    for c in ch:
        if c>4:
            individual[c] += individual[c]+int(individual[c]*random.uniform(-MAX_ALTERATION,MAX_ALTERATION))
        else:
            individual[c] += individual[c]+(individual[c]*random.uniform(-MAX_ALTERATION,MAX_ALTERATION))
            if individual[c]>=1.0:
                individual[c] = 0.9999
            elif individual[c]<=0.0:
                individual[c] = 0.0001
    return individual,

#Mate two parameter sets. Offspring takes after one parent but with crossed over reproduction rates and energy gains
def mate(c,p1,p2):
    count = 0
    for e in range(len(p1)):
        if count>2 and count<7:
            if count<5:
                p1[e] = ((p1[e]+p2[e])/2)
            else:
                p1[e] = (int((p1[e]+p2[e])/2))
        else:
            parent = random.uniform(0,1)
            if parent>=c:
                p1[e] = p2[e]
    return






#--------------------PERFORM GA Functions----------------#
#Run the GA and set up all deap variables necessary
def pp1():
    #pry, prd, g, pyr, pdr, pye, pde, gr = readPP()
    global curr_pop 
    global toolbox
    global s1,s2,s3
    global hof
    global start_time
    start_time = datetime.datetime.now()
    curr_pop = 0
    #Initialise creator variables
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, 1.0,))
    #creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    #Initialise toolbox variables
    toolbox.register("selectParents", selectParents, toolbox)
    toolbox.register("mutate", mutate, toolbox)
    toolbox.register("mate", mate, 0.5)
    toolbox.register("evaluate", evalPP)
    toolbox.register("individual", initIndividual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    s1 = tools.Statistics(lambda ind: ind.fitness.values[0])
    s1.register("c-avg", np.mean)
    s1.register("c-std", np.std)
    s1.register("c-min", np.min)
    s1.register("c-max", np.max)
    s2 = tools.Statistics(lambda ind: ind.fitness.values[1])
    s2.register("p-avg", np.mean)
    s2.register("p-std", np.std)
    s2.register("p-min", np.min)
    s2.register("p-max", np.max)
    s3 = tools.Statistics(lambda ind: ind.fitness.values[2])
    s3.register("n-avg", np.mean)
    s3.register("n-std", np.std)
    s3.register("n-min", np.min)
    s3.register("n-max", np.max)
    pop, log = predprey()
    return pop, log
    
#Run GA
def predprey():
    #Initialise logbook
    global crossover
    global maxevals
    global start_time
    global ga_file
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (s1.fields+s2.fields+s3.fields if s1 and s2 and s3 else [])
    print("Running for: ",maxevals," evaluations")
    ga_file = open(str(MU)+"+"+str(LAMBDA)+"_ga_generations.csv","w")
    ga_file.write("Generation,0,mu,"+str(MU)+",lambda,"+str(LAMBDA)+",seed,"+str(seed)+"\n")
    cont = 1
    #Initialise individuals fitness
    population = toolbox.population(n=MU)
    eval_count = len(population)
    print("Evaluating initial population")
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
#        if ind.fitness.values[2]>0.75:
#            cont = 0
    b = bestFitness(population)
    ga_file.write("parameters,")
    for i in b:
        ga_file.write(str(i)+",")
    ga_file.write("fitnesses,")
    for f in b.fitness.values:
        ga_file.write(str(f)+",")
    ga_file.write("\n")
    ga_file.close()
    print("Starting GA") 
    gen = 0
    log(logbook, population, gen, len(population))
    
    #Start generational process
    while(cont==1):
        gen += 1
        nevals = 0
        ga_file = open(str(MU)+"+"+str(LAMBDA)+"_ga_generations.csv","a")
        ga_file.write("Generation,"+str(gen)+"\n")
        #Generate offspring
        offspring = []

        #If crossover is being used, it is done before mutation
        if crossover:
            for i in range(LAMBDA):
                new = random.uniform(0,1)
                if new<NEW_OFFSPRING:
                    p1 = toolbox.individual()
                else:
                    p1, p2 = [toolbox.clone(x) for x in toolbox.selectParents(population, 2)]
                    toolbox.mate(p1, p2)
                offspring += [p1]
        else:
            offspring = [toolbox.selectParents(population, 1)[0] for i in range(LAMBDA)]
        #Mutate
        for off in offspring:
            off, = toolbox.mutate(off)
            nevals += 1
            off.fitness.values = toolbox.evaluate(off)
        
        eval_count += nevals
        # Select the next generation, favouring the offspring in the event of equal fitness values
        population = favourOffspring(population, offspring, MU)
        #Print a report about the current generation
        print("evals performed: ",eval_count," out of ",maxevals)
        if nevals > 0:
            log(logbook, population, gen, nevals)
            #Save to file in case of early exit
        end_time = datetime.datetime.now()
        time_taken = end_time-start_time
        b = bestFitness(population)
        ga_file.write("parameters,")
        if b.fitness.values[0]>60 and b.fitness.values[2]>0.95:
            cont = 0
        for i in b:
            ga_file.write(str(i)+",")
        ga_file.write("fitnesses,")
        for f in b.fitness.values:
            ga_file.write(str(f)+",")
        ga_file.write("\n")
        ga_file.close()
        f = open(SAVEPATH+"results.txt","w")
        f.write("Random seed: "+str(seed)+"\n")
        f.write("Start time: "+str(start_time)+"\n")
        f.write("End time: "+str(end_time)+"\n")
        f.write("Total GA time: "+str(time_taken)+"\n")
        f.write("Evaluations: "+str(eval_count)+"\nCurrent Population:\n")
        for p in population:
            f.write("\tindividual: "+str(p)+", fitness: "+str(p.fitness.values)+"\n")
        f.write(str(logbook))
    return population, logbook
    
#Run and print
population, logbook = pp1()
print("Final population")
for p in population:
    print("Params",p,"Fitness",p.fitness.values)
print("Logbook")
print(logbook)
f = open(SAVEPATH+"results.txt","r")
r = f.readlines()
f.close()
n = open("ga_results/results_archive/"+GATYPE+"/results.txt","w")
n.writelines(r)
n.close()