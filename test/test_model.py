
'''
Created on Jul 26, 2021

@author: mhindle
'''
from bayesnet.model import BayesPedigreeNetworkModel, _default_alleleprobs, _default_mendelprobs
from pedigree.pedigree_dag import PedigreeDAG
import pandas as pd
import unittest
from pgmpy.inference import VariableElimination, BeliefPropagation
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import mgzip
import cloudpickle, pickle

class TestPedigreeDAG(unittest.TestCase):
    
        def testme(self):
            pedigree = PedigreeDAG.from_file("/gensys/mhindle/quick_sim/poc3/allPed-sex.txt")
            
            print("build model")
            model = BayesPedigreeNetworkModel.generateModel(pedigree,_default_alleleprobs, _default_mendelprobs)
     
            
            # 3929193481809 3893151131942 3893157411622 2
            # 3929193481810 3893151131942 3893157411622 2
            # 3929193481811 3893151131942 3893157411622 2
            # 3929193481812 3893151131942 3893157411622 2
            # 3929193481819 3893151131942 3893157411622 2

            
            infer = VariableElimination(model)
            print("build inference")
            #infer = BeliefPropagation(model.get_markov_blanket('3893151131942'))
            #with mgzip.open('/home/mhindle/eclipse-workspace/fixGen/model.pickle.gz', "wb", thread=12) as act_pickle:
            #            cloudpickle.dump(infer, act_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            #print("calibrate")
            #model.max_calibrate()
            #with mgzip.open('/home/mhindle/eclipse-workspace/fixGen/max_calibrated_model.pickle.gz', "wb", thread=12) as act_pickle:
            #           cloudpickle.dump(infer, act_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            
            sim_data = pd.read_csv("/home/mhindle/simulation_correction/simulatedgenome.ssv", sep=" ", header=0, index_col=0)
            
            all_values = dict(zip(map(str,sim_data.index), sim_data.iloc[:,1].values))
            
            kids_ids = ["3929193481809", "3929193481810", "3929193481811", "3929193481812", "3929193481819"]
            parents_ids = ['3893151131942','3893157411622']
            print("kids: "+str({x:y for x,y in all_values.items() if x in kids_ids}))
            print("parents: "+str({x:y for x,y in all_values.items() if x in parents_ids}))
            #prediction = infer.query(kids_ids,
            #                        evidence={x:y for x,y in all_values.items() if x not in kids_ids}, show_progress=False, joint=True)
            #print(prediction)
            #print(prediction.values)
            
            #infer = VariableElimination(model)
            
            prediction = infer.query(parents_ids,
                                     evidence={x:y for x,y in all_values.items() if x not in parents_ids}, show_progress=False, joint=True)
            
            print("sire parents: "+str({x:y for x,y in all_values.items() if x in ["3851115101322","3851115121506"]}))
            print("dam parents: "+str({x:y for x,y in all_values.items() if x in ["3857121491018","3857115111446"]}))
            
            print(prediction)
            
            smallerregion = kids_ids+parents_ids
            smallerregion.extend(pedigree.get_parents_depth(list(map(int,parents_ids)),2))
            smallerregion.extend(pedigree.get_kids_depth(list(map(int,kids_ids)),2))
            smallerregion = set(smallerregion)
            print(smallerregion)
            
            print([str(node) for node in model.nodes])
            plt.figure(figsize=(8, 8))
            model = model.subgraph(list(map(str,smallerregion))).copy()
            model2 = nx.relabel_nodes(model, {k:"%s (%s) " % (k,v) for k,v in all_values.items() if k in model.nodes}, True)
            pos=graphviz_layout(model2, prog='dot', args='-s250' )
            nx.draw(model2, pos, with_labels=False, arrows=True,node_size=30,font_size=7,
                    node_color=["red" if str(node) in kids_ids else "blue" for node in model2.nodes])
            text = nx.draw_networkx_labels(model2, pos)
            for _,t in text.items():
                t.set_rotation(45)
            plt.savefig("/home/mhindle/simulation_correction/big_example.png")
        
            #print(prediction.values)
            #2 2 2 1 2

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()