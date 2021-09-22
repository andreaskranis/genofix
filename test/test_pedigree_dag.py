'''
Created on Jul 26, 2021

@author: mhindle
'''
from pedigree.pedigree_dag import PedigreeDAG

import unittest
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

class TestPedigreeDAG(unittest.TestCase):
    
    def testme(self):
        pedigree = PedigreeDAG.from_file("/gensys/mhindle/quick_sim/poc3/allPed-sex.txt")
        subgraph = pedigree.subgraph(pedigree.get_decendents(3788146172147))
        
        #pos=graphviz_layout(subgraph, prog='dot')
        #nx.draw(subgraph, pos, with_labels=False, arrows=False)
        #plt.savefig("/home/mhindle/path.pdf")
        
        sire, dam = pedigree.get_parents(3788146191054)
        print("sire %s dam %s " % (sire, dam))
        self.assertEqual(sire, 3746107111635)
        self.assertEqual(dam, 3746101701052)
        
        sire, dam = pedigree.get_parents(3743113491101)
        print("sire %s dam %s " % (sire, dam))
        self.assertEqual(sire, None)
        
        self.assertEqual(pedigree.get_relationship(3779143521305,3809179231803), "daughter")
        self.assertEqual(pedigree.get_relationship(3809179231803,3779143521305), "dam")
        self.assertEqual(pedigree.get_relationship(3779143521305,3848109802129), "granddaughter")
        self.assertEqual(pedigree.get_relationship(3779143521305,3779143501120), "husband")
        
        decendA = pedigree.get_decendents(3788146172147)
        left_commontree = pedigree.get_left_common_tree(3788146172147,[3785146191219])
        
        self.assertEqual(sum([not x in left_commontree for x in decendA]),0)
        self.assertEqual(len(left_commontree) > len(decendA),0)
            
        print(pedigree.get_all_siblings(3809179231803))
        print(pedigree.get_full_siblings(3890148892054))              
        self.assertTrue(len(pedigree.get_full_siblings(3890148892054)) == 1)
        #self.assertEqual()
        #print(pedigree.get_decendents(3788146211035))
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()