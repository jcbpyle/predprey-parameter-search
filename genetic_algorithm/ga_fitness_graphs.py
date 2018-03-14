# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:15:54 2018

@author: James
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = os.getcwd()+"/ga_results/results_archive/"

sns.set_context(rc={"lines.linewidth": 1})
sns.set_style("whitegrid")
#fig = plt.figure(figsize=(24, 8))
#ax = fig.add_subplot(111,projection='3d')
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

drawgraphs()