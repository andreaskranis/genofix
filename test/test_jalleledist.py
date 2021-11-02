'''
Created on Sep 17, 2021

@author: mhindle
'''
import unittest
import numpy as np
import pandas as pd
from empirical.jalleledist3 import JointAllellicDistribution
from scipy.stats import rankdata

class Test(unittest.TestCase):

    def test_basicuse(self):
        sim_data = pd.read_csv("../examples/simulate/simulated_genotype_genome.ssv.gz", sep=" ", 
                               header=0, index_col=0, engine="c", low_memory=False, memory_map=True)
        sim_data = sim_data.iloc[:, :60]
        print(sim_data.shape)
        emp = JointAllellicDistribution(list(sim_data.columns), surround_size=3)
        print("count ")
        emp.countJointFrqAll(sim_data)
        print("done ")
        #print(emp.frequency)
        #print(emp.n_observations)
         
        print(emp.getWindow('SNP50'))
        #print(sim_data.loc[3662,emp.getWindow('SNP50')])
        x = sim_data.loc[3662,emp.getWindow('SNP50')]
        print(x[x.index != 'AX-75205428'].values)
        observedstates = sim_data.loc[3662,emp.getWindow('SNP50')]
        print(observedstates)
        print(sim_data.loc[3662,('SNP50')])
        state_obs = list(emp.getCountTable(observedstates, 'SNP50'))
        print(state_obs)
        prob_states_normalised =  np.divide(state_obs,np.sum(state_obs))
        print(prob_states_normalised)
        rank = rankdata(1-prob_states_normalised, method='max').reshape(prob_states_normalised.shape)
        print(rank)                   
        #[0.26262626 0.52525253 0.21212121] jalleledist2
        #[0.26262626 0.52525253 0.21212121] jalleledist
        observedstates["SNP48"] = 9
        state_obs = list(emp.getCountTable(observedstates, 'SNP50'))
        print(sim_data.loc[3662,('SNP50')])
        print(state_obs)
        prob_states_normalised =  np.divide(state_obs,np.sum(state_obs))
        print(prob_states_normalised)
        rank = rankdata(1-prob_states_normalised, method='max').reshape(prob_states_normalised.shape)
        print(rank)    
        
        print(emp.getObservations('SNP50'))
        
        #[0.26262626 0.52525253 0.21212121] jalleledist2
        #[0.26262626 0.52525253 0.21212121] jalleledist

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
