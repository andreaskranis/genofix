'''
Created on 19 Sep 2022

@author: mhindle
'''
import linecache
import gzip
import numpy as np

class GensCache():
    
    def __init__(self, gensfile, header=False, delimiter=' '):
        self.headerline = None
        self.all_ids = None
        self.kid2index = None
        self.gensfile = gensfile
        self.snps = None
        self.all_ids = None
        if gensfile.endswith(".gz") :
            with gzip.open(gensfile,'r') as filin:
                if header: 
                    self.snps = filin.readline().split(delimiter)[1:]
                self.kid2index = {int(line.rstrip('\n').split(delimiter)[0]):i for i,line in enumerate(filin)}
                self.all_ids = list(self.kid2index.keys())
        else:
            with open(gensfile,'r') as filin:
                if header: 
                    self.snps = filin.readline().split(delimiter)[1:]
                #print(filin.readline())
                self.kid2index = {int(line.rstrip('\n').split(delimiter)[0]):i for i,line in enumerate(filin)}
                self.all_ids = list(self.kid2index.keys())
        print(np.array(self.all_ids))
    
    def getMatrix(self, ids) :
        #print("getMatrix")
        #print(np.array(ids))
        matresult = list()
        ids=[self.kid2index[int(x)]+3 for x in ids if int(x) in self.kid2index]
        errors = [x for x in ids if int(x) not in self.kid2index]
        #if len(errors) > 0:
        #   print("ERROR: following ids not found: %s : ERRORS" % ", ".join(map(str,errors)))
        
        get_values = lambda idA: np.array(linecache.getline(self.gensfile, idA).rstrip().split(' ')[1:], dtype=np.uint8)
        matresult = list(map(get_values, ids))
        return(np.vstack(matresult))

g = GensCache("C:/Users/mhindle/Documents/psuk_1_00", header=True)
print(g.getMatrix(["3653120310902","3656123492012","3659126571902","3662126631109"]))
print(g.getMatrix(["3653120310902","3656123492012","3659126571902","3662126631109"]).shape)


