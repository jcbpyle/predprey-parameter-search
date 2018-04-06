# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:15:54 2018

@author: James
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = os.getcwd()+"/ga_results/mavericks_results/"

sns.set_context(rc={"lines.linewidth": 2})
sns.set_style("whitegrid")
#fig = plt.figure(figsize=(24, 8))
#ax = fig.add_subplot(111,projection='3d')

def fitness_graph(g):
    gcount = 0
    f = open(RESULTS_PATH+"20+1_ga_generations.csv","r")
    fitnesses = []
    f0 = []
    f1 = []
    f2 = []
#        f00 = []
#        f01 = []
#        f10 = []
#        f11 = []
    for l in f:
        if gcount<g:
            sp = l.split(",")
            if sp[0]=="Generation":
                gcount+=1
            else:
#                    if sp[13]<0.5:
#                        f00.append([sp[11]])
#                        f10.append([sp[12]])
#                    else:
#                        f01.append([sp[11]])
#                        f11.append([sp[12]])
                #fitnesses.append([gcount,sp[11],sp[12],sp[13]])
                f0.append([gcount,float(sp[11])])
                f1.append([gcount,float(sp[12])])
                f2.append([gcount,float(sp[13])])
    plt_fit([f[0] for f in f0], [f[1] for f in f0],"Fitness 1 over GA generations","Fitness_1")
    plt_fit([f[0] for f in f1], [f[1] for f in f1],"Fitness 2 over GA generations","Fitness_2")
    plt_fit([f[0] for f in f2], [f[1] for f in f2],"Fitness 3 over GA generations","Fitness_3")
    #plt_fit([f[0] for f in fitnesses], [f[2] for f in fitnesses],"Fitness 2 over GA generations","fitness2")
    #plt_fit([f[0] for f in fitnesses], [f[3] for f in fitnesses],"Fitness 3 over GA generations","fitness3")
   
#    plt.figure(figsize=(12, 6))
#    plt.title("Fitness 2 over GA generations", fontsize=24, color='black')
#    plt.xlabel("Iteration", fontsize=18)
#    plt.ylabel("Fitness", fontsize=16)
#    plt.yticks(fontsize=12)
#    #plt.xlim(0, 10)
#    #plt.ylim(0, 10000)
#    plt.xticks(fontsize=12)
#    plt.plot([f[0] for f in fitnesses], [f[2] for f in fitnesses], 'bo', label="Fitness 2")
#    plt.legend(frameon=True, fontsize=18, loc=2)
#    #plt.tight_layout()
#    plt.savefig(os.getcwd()+"/fitness2.png", bbox_inches='tight')
#    plt.savefig(os.getcwd()+"/fitness2.pdf", bbox_inches='tight')
#    plt.show()
#    plt.close()
#    plt.figure(figsize=(12, 6))
#    plt.title("Fitness 3 over GA generations", fontsize=24, color='black')
#    plt.xlabel("Iteration", fontsize=18)
#    plt.ylabel("Fitness", fontsize=16)
#    plt.yticks(fontsize=12)
#    #plt.xlim(0, 10)
#    #plt.ylim(0, 10000)
#    plt.xticks(fontsize=12)
#    plt.plot([f[0] for f in fitnesses], [f[3] for f in fitnesses], 'bo', label="Fitness 3")
#    plt.legend(frameon=True, fontsize=18, loc=2)
#    #plt.tight_layout()
#    plt.savefig(os.getcwd()+"/fitness3.png", bbox_inches='tight')
#    plt.savefig(os.getcwd()+"/fitness3.pdf", bbox_inches='tight')
#    plt.show()
#    plt.close()
    return

def plt_fit(x,y,title,save):
    plt.figure(figsize=(10, 4))
    plt.title(title, fontsize=24, color='black')
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Fitness", fontsize=18)
    plt.yticks(fontsize=14)
    #plt.xlim(0, 5001)
    #plt.ylim(0, 60)
    plt.xticks(fontsize=14)
#    for i in range(len(x)):
#        plt.plot(x[i],y[i], 'b-')
    plt.plot(x,y, 'b-')
    plt.legend(frameon=True, fontsize=18, loc=2)
    plt.tight_layout()
    plt.savefig(os.getcwd()+"/"+save+".png", bbox_inches='tight')
    plt.savefig(os.getcwd()+"/"+save+".pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

def population_graph():
    
    return

fitness_graph(5000)
def drawgraphs():
    directories = [x[0] for x in os.walk(RESULTS_PATH)]
    fcount=0
    for folder in directories:
        print(folder)
        title = folder.split("/")[3]
        mu = title.split("+")[0]
        fig = plt.figure(figsize=(24, 8))
        fig.suptitle(title, fontsize=16, color="black")
        if os.path.exists(folder+"/results.txt"):
            res = open(folder+"/results.txt")
            lcount=0
            dc = pd.DataFrame({'gen':[],'crossover':[]})
            dp = pd.DataFrame({'gen':[],'population_difference':[]})
            dn = pd.DataFrame({'gen':[],'non_zero':[]})
            start = int(mu)+7
            for li in res:
                s = li.split()
#                if lcount==start:
#                    print(s)
                if lcount>=start:
                    d1 = {'gen':[int(s[0])],'crossover':[float(s[2])]}
                    d2 = {'gen':[int(s[0])],'population_difference':[float(s[6])]}
                    d3 = {'gen':[int(s[0])],'non_zero':[float(s[10])]}
                    ndf1 = pd.DataFrame(data=d1)
                    ndf2 = pd.DataFrame(data=d2)
                    ndf3 = pd.DataFrame(data=d3)
                    dc = pd.concat([dc,ndf1],axis=0)
                    dp = pd.concat([dp,ndf2],axis=0)
                    dn = pd.concat([dn,ndf3],axis=0)
                lcount+=1
                
            (ax1,ax2,ax3) = fig.subplots(1,3)
            ax1.plot(dc['gen'], dc['crossover'])
            ax2.plot(dp['gen'], dp['population_difference'])
            ax3.plot(dn['gen'], dn['non_zero'])
            ax1.set_title('Crossover', fontsize=18, color='black')
            ax2.set_title('Population Difference', fontsize=18, color='black')
            ax3.set_title('Non Zero', fontsize=18, color='black')
            ax1.set_xlabel("GA Generation", fontsize=18)
            ax1.set_ylabel("Fitness", fontsize=18)
            ax2.set_xlabel("GA Generation", fontsize=18)
            ax2.set_ylabel("Fitness", fontsize=18)
            ax3.set_xlabel("GA Generation", fontsize=18)
            ax3.set_ylabel("Fitness", fontsize=18)
            #plt.savefig(PATH+"/ga_results/graphs/"+title+".svg")
            #Save image
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            fig.savefig(RESULTS_PATH+"/"+title+".png", bbox_inches='tight')
            
        fcount+=1
    return

#drawgraphs()