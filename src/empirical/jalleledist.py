'''
Created on Sep 6, 2021

@author: mhindle
'''
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict, Union, Set
import itertools
from collections import defaultdict

class JointAllellicDistribution(object):

    def __init__(self, pseudocount = 1, window_size=3):
        self.pseudocount = pseudocount
        self.frequency: Dict[Tuple[str,int],Dict[Tuple[str,int],Dict[Tuple[str,int],int]]] = dict()
        self.n_observations: Dict[Tuple[str,str,str]] = defaultdict(int)
        self.window_size = window_size

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
