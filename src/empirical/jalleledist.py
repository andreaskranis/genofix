'''
Created on Sep 6, 2021

@author: mhindle
'''
import numpy as np
import pandas as pd
import numbers
from typing import Tuple, List, Dict, Union, Set
import itertools
from collections import defaultdict

class JointAllellicDistribution(object):

    def __init__(self, snp_ordered, pseudocount = 1, surround_size=1):
        self.pseudocount = pseudocount
        self.frequency: Dict[Tuple[str,int],Dict[Tuple[str,int],Dict[Tuple[str,int],int]]] = dict()
        self.n_observations: Dict[Tuple[str,str,str]] = defaultdict(int)
        self.surround_size = surround_size
        self.window_size = (surround_size*2)+1
        self.snp_ordered = snp_ordered
    
    def getWindow(self, targetSnp):
        targetpos = self.snp_ordered.index(targetSnp)
        startpos_snp = targetpos-self.surround_size
        endpos_snp = targetpos+self.surround_size+1
        return(self.snp_ordered[startpos_snp:endpos_snp])
    
    def getCountTable(self, observedstates, targetSnp):
        targetpos = self.snp_ordered.index(targetSnp)
        all_obs = [(snpid,observedstates[snpid]) for snpid in self.getWindow(targetSnp)]
        
        def copypastefunc(x):
            r = all_obs.copy()
            r[self.surround_size] = (targetSnp, x)
            return(r)
        
        for state, query in enumerate(list(map(copypastefunc, [0,1,2]))):
            #print("%s == %s" % (state, query))
            workinghash = self.frequency
            for item in query:
                workinghash = workinghash[item]
            if isinstance(workinghash, numbers.Number):
                yield workinghash #it should be the result
            else:
                raise Exception("incomplete traversal of nested hash: final %s state %s" % (workinghash, state))

    def countJointFrq(self, table, mask):
        subset = table.iloc[mask.all(axis=1),:]
        column_names = list(subset.columns)
        
        for values in list(itertools.product([0,1,2], repeat=self.window_size)):
            conditions = zip(column_names, values)
            rows_that_meet = np.logical_and.reduce([subset[snp] == value for snp,value in conditions])
            keys = list(zip(column_names, values))
            obs = np.count_nonzero(rows_that_meet)
            self.recurse_set_dict(self.frequency, keys, obs)
            self.n_observations[tuple(column_names)] += (obs+self.pseudocount) # this we keep track of how many observations there have been for these three snps
    
    
    def recurse_set_dict(self, d, queue, value):
        f = queue.pop(0)
        if len(queue) > 0:
            if f not in d:
                d[f] = dict()
                self.recurse_set_dict(d[f], queue, value)
            else:
                self.recurse_set_dict(d[f], queue, value)
        else:
            d[f] = (value+self.pseudocount) # we record the observations for this state combo
            
    def countJointFrqAll(self, table):
        for i in range(table.shape[1]-self.window_size+1):
            snp_window = table.columns.tolist()[i:i+self.window_size]
            mask = np.ones((table.shape[0],3),dtype=bool)
            self.countJointFrq(table.loc[:,snp_window], mask)
            
                

#===============================================================================
# sim_data = pd.read_csv("/home/mhindle/simulation_correction_newalgo1/simulatedgenome.ssv", sep=" ", header=0, index_col=0)
# 
# emp = JointAllellicDistribution()
# emp.countJointFrqAll(sim_data)
# 
# print(emp.frequency)
# print(emp.n_observations)
#===============================================================================
#===============================================================================
# sim_data = pd.read_csv("/home/mhindle/simulation_correction_newalgo1/simulatedgenome.ssv", sep=" ", header=0, index_col=0)
# print(sim_data)
# 
# emp = JointAllellicDistribution(list(sim_data.columns), surround_size=2)
# emp.countJointFrqAll(sim_data)
#  
# print(emp.frequency)
# print(emp.n_observations)
# 
# print(emp.getWindow('AX-75205428'))
# print(sim_data.loc[4004165481949,emp.getWindow('AX-75205428')])
# 
# print(list(emp.getCountTable(sim_data.loc[4004165481949,emp.getWindow('AX-75205428')], 'AX-75205428')))
#===============================================================================

