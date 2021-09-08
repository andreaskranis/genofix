'''
Created on Jul 26, 2021

@author: mhindle
'''
import pedigree
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Union, Any, Optional, Set, Generator
import networkx.algorithms.dag as dag

class PedigreeDAG(nx.DiGraph):
    '''
    classdocs
    '''
    
    def balance_nodes(self, nodes):
        found = 0
        for node in [n for n in nodes if n in self.kid2sire.keys() or n in self.kid2dam.keys()]:
                sire = self.kid2sire[node]
                dam = self.kid2dam[node]
                if sire in nodes and dam not in nodes:
                    nodes.append(dam)
                    found+=1
                    #print("added d:%s with s:%s for k:%s " % (dam, sire, node))
                if dam in nodes and sire not in nodes:
                    nodes.append(sire)
                    found+=1
                    #print("added s:%s with d:%s for k:%s " % (sire, dam, node))
        if found > 0:
            nodes = self.balance_nodes(nodes)
        return nodes
    
    def get_subset(self, nodes:List, balance_parents=True):
        
        if balance_parents:
            nodes = self.balance_nodes(nodes)
        
        dag = PedigreeDAG()
        dag.males = set([x for x in self.males if x in nodes])
        dag.females = set([x for x in self.females if x in nodes])
        dag.kid2sire = {k:s for k,s in self.kid2sire.items() if (k in nodes and s in nodes) }
        dag.kid2dam = {k:d for k,d in self.kid2dam.items() if (k in nodes and d in nodes)}
        dag.sire2kid = PedigreeDAG._indexone2many([(int(sire),int(kid)) for kid, sire in self.kid2sire.items()])
        dag.dam2kid = PedigreeDAG._indexone2many([(int(dam),int(kid)) for kid, dam in self.kid2dam.items()])
        
        dag.generationIndex = defaultdict(set)
        for kid in dag.males.union(dag.females):
            generation = min([PedigreeDAG.recursiveDepth(kid, dag.kid2sire), PedigreeDAG.recursiveDepth(kid, dag.kid2dam)])
            dag.generationIndex[generation].add( kid )
        
        for male in dag.males:
            dag.add_node(male, sex=1)
        for female in dag.females:
            dag.add_node(female, sex=2)
        for kid,sire in dag.kid2sire.items():
            dag.add_edge(sire, kid)
        for kid,dam in dag.kid2dam.items():
            dag.add_edge(dam , kid)
        
        return dag
    
    @staticmethod
    def recursiveDepth(current, dictionary, count=0) :
        if current in dictionary.keys():
            return(PedigreeDAG.recursiveDepth(dictionary[current], dictionary, count+1))
        else:
            return(count)
    
    @staticmethod
    def _indexone2many(itemsIn: Any) :
        d = defaultdict(set)
        for key, value in itemsIn:
            d[key].add(value)
        return d

    def get_nodes(self):
        return(list(self.males)+list(self.females))

    @staticmethod
    def from_table(pedigree: Any):
        dag = PedigreeDAG()
        dag.males = set([int(sire) for kid, sire, dam, sex in pedigree]+[kid for kid, sire, dam, sex in pedigree if sex == 1])
        dag.females = set([int(dam) for kid, sire, dam, sex in pedigree]+[kid for kid, sire, dam, sex in pedigree if sex == 2])
        dag.kid2sire = {int(kid):int(sire) for kid, sire, dam, sex in pedigree}
        dag.kid2dam = {int(kid):int(dam) for kid, sire, dam, sex in pedigree}
        dag.sire2kid = PedigreeDAG._indexone2many([(int(sire),int(kid)) for kid, sire, dam, sex in pedigree])
        dag.dam2kid = PedigreeDAG._indexone2many([(int(dam),int(kid)) for kid, sire, dam, sex in pedigree])
        
        dag.generationIndex = defaultdict(set)
        for kid in dag.males.union(dag.females):
            generation = min([PedigreeDAG.recursiveDepth(kid, dag.kid2sire), PedigreeDAG.recursiveDepth(kid, dag.kid2dam)])
            dag.generationIndex[generation].add( kid )
        
        for line in pedigree:
            kid, sire, dam, sex = map(int,line)
            dag.add_node(kid, sex=sex)
            if not dag.has_node(sire):
                dag.add_node(sire, sex=1)
            if not dag.has_node(dam):
                dag.add_node(dam, sex=2)
            dag.add_edge(sire, kid)
            dag.add_edge(dam, kid)
        return dag
    
    @staticmethod
    def from_file(file:str):
        pedigree = np.loadtxt(file, delimiter=' ', dtype=int)
        return(PedigreeDAG.from_table(pedigree))
    
    def get_parents_depth(self, kids:List[Union[int,str]], depth:int) -> Generator[Union[int,str], None, None]:
        if depth > 0 and len(kids) > 0:
            new = np.concatenate([[p for p in self.get_parents(kid) if p is not None] for kid in kids])
            yield from new
            yield from self.get_parents_depth(new, depth-1)
    
    def get_kids_depth(self, kids:List[Union[int,str]], depth:int) -> Generator[Union[int,str], None, None]:
        if depth > 0 and len(kids) > 0:
            new = np.concatenate([[p for p in self.get_kids(kid) if p is not None] for kid in kids])
            yield from new
            yield from self.get_kids_depth(new, depth-1)
    
    def get_parents(self, kid: Union[int,str]) -> Tuple[Optional[int],Optional[int]]:
        '''
        return tuple of ids (sire, dam) or None as id if unknown
        '''
        if kid not in self:
            return (None, None)
        parents = list(self.predecessors(kid))
        if len(parents) == 0:
            return (None, None)
        elif len(parents) == 1:
            if parents[0] in self.males:
                return (parents[0], None)
            elif parents[0] in self.females:
                return (None, parents[0])
            else:
                raise Exception("Malformed pedigree DAG")
        else:
            p1, p2 = parents
            if p1 in self.males and p2 in self.females:
                return(p1,p2)
            elif p1 in self.females and p2 in self.males:
                return(p2,p1)
            else:
                raise Exception("Malformed pedigree DAG")

    def get_kids(self, parent:int) -> List[Union[int,str]]:
        '''
        return list of kid ids
        '''
        return(self.successors(parent))
    
    def get_all_siblings(self, kid:int) -> List[Union[int,str]]:
        sire, dam = self.get_parents(kid)
        sibs = set()
        if sire != None :
            sibs.update(self.get_kids(sire))
        if dam != None :
            sibs.update(self.get_kids(dam))
        if len(sibs) > 0 :
            sibs.remove(kid)
        return(list(sibs))
    
    def get_full_siblings(self, kid:int) -> List[Union[int,str]]:
        sire, dam = self.get_parents(kid)
        sibs = set()
        if sire != None and dam != None:
            sibs = set(self.get_kids(sire)).intersection(set(self.get_kids(dam)))
        if len(sibs) > 0 :
            sibs.remove(kid)
        return(list(sibs))
    
    def get_ancestors(self, parent:int) -> List[Union[int,str]]:
        '''
        return list of kid ids
        '''
        return(dag.ancestors(self, parent))
    
    def get_decendents(self, parent:int) ->List[Union[int,str]]:
        '''
        return list of kid ids
        '''
        return(dag.descendants(self, parent))
    
    def get_left_common_tree(self, a:int, bs:List[int]) -> List[Union[int,str]]:
        left_tree = set(self.get_decendents(a))
        for b in bs:
            common_decendents = np.intersect1d(list(self.get_decendents(a)),list(self.get_decendents(b)))
            #print(self.get_decendents(a))
            #print(self.get_decendents(b))
            #print(common_decendents)
            for x in common_decendents:
                for path in nx.all_simple_paths(self, x,b):
                    left_tree.update(path)
        return(list(left_tree))
    
    def get_partners(self, parent:int) -> List[Union[int,str]]:
        '''
        return list of kid ids
        '''
        parents: Set[int] = set(np.concatenate([self.get_parents(kid) for kid in self.get_kids(parent)]))
        if len(parents) > 0 :
            parents.remove(parent)
        return(list(parents))
    
    def get_grandkids(self, parent:int) -> List[Union[int,str]] :
        '''
        return list of kid ids
        '''
        kids = list(self.successors(parent))
        if (len(kids) > 0) :
            return(np.concatenate([list(self.successors(kid)) for kid in kids]))
        else:
            return([])
    
    def get_relationship(self, kid, other):
        '''
        helpful for getting human readable pedigree relations for debugging etc...
        '''
        if other in self.get_kids(kid):
            return("son" if other in self.males else "daughter")
        elif other in self.get_parents(kid):
            return("sire" if other in self.males else "dam")
        elif other in self.get_grandkids(kid):
            return("grandson" if other in self.males else "granddaughter")
        elif kid in self.get_grandkids(other):
            return("grandfather" if other in self.males else "grandmother")
        elif other in self.get_partners(kid):
            return("husband" if other in self.males else "wife")
        else:
            return("other")
    
    def as_ped(self):
        '''
        to standard pedigree format
        '''
        lines = list()
        for _generation, kids in self.generationIndex.items():
            for kid in kids:
                if kid in self.kid2sire and kid in self.kid2dam:
                    lines.append((kid, self.kid2sire[kid],self.kid2dam[kid], 1 if kid in self.males else 2))
        return(lines)
        
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
        
