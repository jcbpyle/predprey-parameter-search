# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:47:16 2018

@author: James
"""

import random
import queue
import os
import threading
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import matplotlib.pyplot as plt
import datetime
#import seaborn as sns

parser = argparse.ArgumentParser(description='Python implementation of benchmarking simulations using multiple GPUs')
parser.add_argument('inputs', metavar='input_file', type=str, nargs='+',
                   help='The input parameter sets to simulate in batch')
parser.add_argument('parameters', metavar='num_params', type=int, nargs='+',
                   help='The number of parameter sets to generate')
args = parser.parse_args()
f = args.inputs[0]
n = args.parameters[0]

GPUS_AVAILABLE = cuda.Device(0).count()
SAVEPATH = os.getcwd()+"/temp/"

#sns.set_context(rc={"lines.linewidth": 1})
#sns.set_style("whitegrid")  

exitQ = 0   
workQueue = None
queueLock = threading.Lock()
        
class benchQueueThread(threading.Thread):
    def __init__(self, tn, device, q):
        threading.Thread.__init__(self)
        self.tn = tn
        self.device = device
        self.q = q
    def run(self):
        threadQueueFunction(self.tn, self.q, self.device)


def threadQueueFunction(tn, q, d):
    global queueLock, exitQ
    #Make directories
    if not os.path.exists(SAVEPATH+str(d)):
        os.makedirs(SAVEPATH+str(d))
    if not os.path.exists(SAVEPATH+str(d)+"/0.xml"):
        open(SAVEPATH+str(d)+"/0.xml", "w").close()
    while exitQ==0:
        queueLock.acquire()
        if not workQueue.empty():
            x = q.get()
            queueLock.release()
            if os.name=='nt':
                genComm = os.getcwd()+"/../xmlGenEx3.exe "+SAVEPATH+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])
                command = os.getcwd()+"/../PreyPredator.exe "+SAVEPATH+str(d)+"/0.xml "+str(1000)+" "+str(d)
            else:
                genComm = "../xmlGenEx3 "+SAVEPATH+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])        
                command = "../PreyPredator_console "+SAVEPATH+str(d)+"/0.xml "+str(1000)+" "+str(d)
            os.system(genComm)
            os.system(command)
        else:
            queueLock.release()
                    
def generateParameterSet():
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

def createFile(f, n):
    csv = open(os.getcwd()+"/"+f+".csv","w")
    for i in range(n):
        x = generateParameterSet()
        for v in x:
            csv.write(str(v)+",")
        csv.write("\n")
    csv.close()
    return

def benchmark(f,n):
    global queueLock, workQueue, exitQ
    createFile(f,n)
    params = open(os.getcwd()+"/"+f+".csv","r")
    X = []
    res = [[],[]]
    workQueue = queue.Queue(n)
    for l in params:
        sp = l.split(",")
        X.append(sp)
    for i in [1,2,4,8]:
        res[0].append(i)
        if (i<=GPUS_AVAILABLE):
            threads = []
            exitQ = 0
            queueLock.acquire()
            for a in X:
                workQueue.put(a)
            queueLock.release()
            start_time = datetime.datetime.now()
            for b in range(i):
                thread = benchQueueThread(i, b,workQueue)
                thread.start()
                threads.append(thread)
            while not workQueue.empty():
                pass
            exitQ = 1
            for t in threads:
                t.join()
                
            end_time = datetime.datetime.now()
            time_taken = end_time-start_time
            res[1].append(time_taken.seconds+(time_taken.microseconds/1000000))
        else:
            res[1].append(0)
    params.close()
    r = open("results.csv","w")
    for i in range(len(res[0])):
        r.write("gpus,"+str(res[0][i])+",time,"+str(res[1][i])+"\n")
    r.close()
    return

def generate_graph():
    r = open("results.csv","r")
    res = [[],[]]
    for l in r:
        sp = l.split(",")
        if not(sp[0]=="num_parameters"):
            res[0].append(int(sp[1]))
            res[1].append(float(sp[3]))
    title = "Batch Simulation Performance with multiple GPUs"
    plt.figure(figsize=(10,6))
    plt.legend(frameon=True, fontsize=18, loc=1)
    plt.title(title, fontsize=24, color='black')
    plt.xlabel("GPUs used", fontsize=18)
    plt.ylabel("Seconds to complete simulations", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    print(res)
    plt.plot(res[0], res[1])
    plt.savefig(os.getcwd()+"/performance_graph.png", bbox_inches='tight')
    plt.savefig(os.getcwd()+"/performance_graph.pdf", bbox_inches='tight')
    return


benchmark(f,n)
#generate_graph()