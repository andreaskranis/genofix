'''
Created on Jul 26, 2021

@author: mhindle
'''
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Any, Optional, Set, Generator
import networkx.algorithms.dag as dag

class PedigreeDAG(nx.DiGraph):
    '''
    classdocs
    '''
    def __init__(self, incoming_graph_data=None):
        super().__init__(incoming_graph_data=incoming_graph_data)
        if incoming_graph_data is not None:
            self.males = incoming_graph_data.males.copy()
            self.females = incoming_graph_data.females.copy()
            self.kid2sire = incoming_graph_data.kid2sire.copy()
            self.kid2dam = incoming_graph_data.kid2dam.copy()
            self.sire2kid = incoming_graph_data.sire2kid.copy()
            self.dam2kid = incoming_graph_data.dam2kid.copy()
            self.generationIndex = incoming_graph_data.generationIndex.copy()
        else:
            self.males = {}
            self.females = {}
            self.kid2sire = set()
            self.kid2dam = set()
            self.sire2kid = set()
            self.dam2kid = set()
            self.generationIndex = defaultdict(set)
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls, incoming_graph_data=self)
        result.__dict__.update(self.__dict__)
        return cls
    
    def split_pedigree(self, min_cluster_size=100):
        partner_pairs = list(set([(sire,dam) for sire,dam in [self.get_parents(kid) for kid in self.males.union(self.females)] if sire != None and dam != None]))
        clustered_pairs = []
        for sire, dam in partner_pairs:
            found = False
            for cluster in clustered_pairs:
                if sire in cluster: 
                    cluster.add(dam)
                    found = True
                    break
                elif dam in cluster:
                    cluster.add(sire)
                    found = True
                    break
            if not found:
                clustered_pairs.append({sire, dam})
        
        pairedkids = set([x[0] for x in partner_pairs]+[x[1] for x in partner_pairs])
        allkids = self.males.union(self.females)
        singlekids = [x for x in allkids if x not in pairedkids]
        for kid in singlekids:
            sire, dam = self.get_parents(kid)
            found = False
            for cluster in clustered_pairs:
                if sire is not None and sire in cluster : 
                    cluster.add(kid)
                    found = True
                    break
                elif dam is not None and dam in cluster:
                    cluster.add(kid)
                    found = True
                    break
            if not found:
                raise Exception("%s illegal lone node in pedigree" % kid)
        
        #print(len(clustered_pairs))
        
        for _i in range(len(clustered_pairs)):
            for cluster in sorted(clustered_pairs, key=lambda x: len(x)):
                #self.kikid2sire
                found = False
                for target in sorted([x for x in clustered_pairs if x != cluster], key=lambda x: len(x)):
                    if len(cluster.intersection(target)) > 0:
                        target.update(cluster)
                        clustered_pairs.remove(cluster)
                        found = True
                        break
                    if found:
                        break
        
        #print(len(clustered_pairs))
           
        for size in range(min_cluster_size):
            for _i in range(len(clustered_pairs)):
                for cluster in [x for x in clustered_pairs if len(x) == size]:
                    sires = set([self.kid2sire[x] for x in cluster if x in self.kid2sire])
                    dams = set([self.kid2dam[x] for x in cluster if x in self.kid2dam])
                    parents = sires.union(dams)
                    found = False
                    for target in sorted([x for x in clustered_pairs if x != cluster], key=lambda x: len(x)):
                        if len(parents.intersection(target)) > 0:
                            target.update(cluster)
                            clustered_pairs.remove(cluster)
                            found = True
                            break
                    if found:
                        break
                    kids = [list(self.sire2kid[x]) for x in cluster if x in self.sire2kid]
                    if len(kids) > 0:
                        kids = set(np.concatenate(kids))
                    kids2 = [list(self.dam2kid[x]) for x in cluster if x in self.dam2kid]
                    if len(kids2) > 0:
                        kids = kids.union(set(np.concatenate(kids2)))
                    for target in sorted([x for x in clustered_pairs if x != cluster], key=lambda x: len(x)):
                        if len(kids.intersection(target)) > 0:
                            target.update(cluster)
                            clustered_pairs.remove(cluster)
                            found = True
                            break
                    if found:
                        break
        return(clustered_pairs)
    
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
        dag.males = set([int(sire) for _kid, sire, _dam, _sex in pedigree if int(sire) > 0]+[int(kid) for kid, sire, _dam, sex in pedigree if sex == 1 and int(kid) > 0])
        dag.females = set([int(dam) for kid, sire, dam, sex in pedigree if int(dam) > 0]+[int(kid) for kid, sire, dam, sex in pedigree if sex == 2  and int(kid) > 0])
        dag.kid2sire = {int(kid):int(sire) for kid, sire, dam, sex in pedigree if int(sire) > 0 and int(kid) > 0}
        dag.kid2dam = {int(kid):int(dam) for kid, sire, dam, sex in pedigree if int(dam) > 0 and int(kid) > 0}
        dag.sire2kid = PedigreeDAG._indexone2many([(int(sire),int(kid)) for kid, sire, dam, sex in pedigree if int(sire) > 0 and int(kid) > 0])
        dag.dam2kid = PedigreeDAG._indexone2many([(int(dam),int(kid)) for kid, sire, dam, sex in pedigree if int(dam) > 0 and int(kid) > 0])
        
        dag.generationIndex = defaultdict(set)
        for kid in dag.males.union(dag.females):
            generation = min([PedigreeDAG.recursiveDepth(kid, dag.kid2sire), PedigreeDAG.recursiveDepth(kid, dag.kid2dam)])
            dag.generationIndex[generation].add( kid )
        
        for line in pedigree:
            kid, sire, dam, sex = map(int,line)
            if kid > 0:
                dag.add_node(kid, sex=sex)
            if not dag.has_node(sire) and sire > 0:
                dag.add_node(sire, sex=1)
            if not dag.has_node(dam) and dam > 0:
                dag.add_node(dam, sex=2)
            if kid > 0 and sire > 0:
                dag.add_edge(sire, kid)
            if kid > 0 and dam > 0:   
                dag.add_edge(dam, kid)
        return dag
    
    @staticmethod
    def from_file(file:str):
        pedigree = np.loadtxt(file, delimiter=' ', dtype=np.int64, usecols = (0,1,2,3))
        return(PedigreeDAG.from_table(pedigree))
    
    def get_parents_depth(self, kids:List[int], depth:int) -> Generator[int, None, None]:
        if isinstance(kids, int) or isinstance(kids, np.int):
            kids = [kids]
        
        if depth > 0 and len(kids) > 0:
            if len(kids) == 1:
                new = [p for p in self.get_parents(kids[0]) if p is not None]
            else:
                new = np.concatenate([[p for p in self.get_parents(kid) if p is not None] for kid in kids])
            yield from new
            yield from self.get_parents_depth(new, depth-1)
    
    def get_kids_depth(self, kids:List[int], depth:int) -> Generator[int, None, None]:
        if isinstance(kids, int) or isinstance(kids, np.int):
            kids = [kids]
        
        if depth > 0 and len(kids) > 0:
            if len(kids) == 1:
                new = [p for p in self.get_kids(kids[0]) if p is not None]
            else:
                new = np.concatenate([[p for p in self.get_kids(kid) if p is not None] for kid in kids])
            yield from new
            yield from self.get_kids_depth(new, depth-1)
    
    def get_parents(self, kid: int) -> Tuple[Optional[int],Optional[int]]:
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

    def get_kids(self, parent:int) -> List[int]:
        '''
        return list of kid ids
        '''
        return(self.successors(parent))
    
    def get_all_siblings(self, kid:int) -> List[int]:
        sire, dam = self.get_parents(kid)
        sibs = set()
        if sire != None :
            sibs.update(self.get_kids(sire)) # type: ignore
        if dam != None :
            sibs.update(self.get_kids(dam)) # type: ignore
        if len(sibs) > 0 :
            sibs.remove(kid)
        return(list(sibs))
    
    def get_full_siblings(self, kid:int) -> List[int]:
        sire, dam = self.get_parents(kid)
        sibs = set()
        if sire != None and dam != None:
            sibs = set(self.get_kids(sire)).intersection(set(self.get_kids(dam))) # type: ignore
        if len(sibs) > 0 :
            sibs.remove(kid)
        return(list(sibs))
    
    def get_ancestors(self, parent:int) -> List[int]:
        '''
        return list of kid ids
        '''
        return(dag.ancestors(self, parent))
    
    def get_decendents(self, parent:int) ->List[int]:
        '''
        return list of kid ids
        '''
        return(dag.descendants(self, parent))
    
    def get_left_common_tree(self, a:int, bs:List[int]) -> List[int]:
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
    
    def get_partners(self, parent:int) -> List[int]:
        '''
        return list of kid ids
        '''
        kids = list(self.get_kids(parent))
        if len(kids) > 0:
            parents: Set[int] = set(np.concatenate([self.get_parents(kid) for kid in kids]))
            if len(parents) > 0 :
                parents.remove(parent)
            return(list(parents))
        else:
            return([])
    
    def get_grandkids(self, parent:int) -> List[int] :
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
