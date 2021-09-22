'''
Created on Sep 17, 2021

@author: mhindle
'''
import unittest
import numpy as np
import pandas as pd
from empirical.jalleledist import JointAllellicDistribution

class Test(unittest.TestCase):

    def testBasicUse(self):
        sim_data = pd.read_csv("/home/mhindle/simulation_correction_newalgo1/simulatedgenome.ssv", sep=" ", header=0, index_col=0)
        emp = JointAllellicDistribution(list(sim_data.columns), surround_size=2)
        emp.countJointFrqAll(sim_data)
          
        print(emp.frequency)
        print(emp.n_observations)
         
        print(emp.getWindow('AX-75205428'))
        print(sim_data.loc[4004165481949,emp.getWindow('AX-75205428')])
        x = sim_data.loc[4004165481949,emp.getWindow('AX-75205428')]
        print(x[x.index != 'AX-75205428'].values)
        state_obs = list(emp.getCountTable(sim_data.loc[4004165481949,emp.getWindow('AX-75205428')], 'AX-75205428'))
        print(state_obs)
        prob_states_normalised =  np.divide(state_obs,np.sum(state_obs))
        print(prob_states_normalised)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()