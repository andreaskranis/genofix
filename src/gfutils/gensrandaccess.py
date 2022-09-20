'''
Created on 19 Sep 2022

@author: mhindle
'''
import linecache
import gzip
import numpy as np

class GensCache():
    
    def __init__(self, gensfile, header=False, delimiter=' '):
        print("bla %s" % gensfile)
        self.headerline = None
        self.all_ids = None
        self.kid2index = None
        self.gensfile = gensfile
        self.snps = None
        self.all_ids = None
        print(gensfile)
        if gensfile.endswith(".gz") :
            with gzip.open(gensfile,'r') as filin:
                if header: 
                    self.snps = filin.readline().split(delimiter)[1:]
                kid2index = {line.rstrip('\n').split(delimiter)[0]:i for i,line in enumerate(filin)}
                self.all_ids = list(kid2index.values())
        else:
            with open(gensfile,'r') as filin:
                if header: 
                    self.snps = filin.readline().split(delimiter)[1:]
                #print(filin.readline())
                self.kid2index = {line.rstrip('\n').split(delimiter)[0]:i for i,line in enumerate(filin)}
                self.all_ids = list(self.kid2index.values())
    
    def getMatrix(self, ids) :
        matresult = list()
        ids=[self.kid2index[x]+3 for x in ids]
        for idA in ids:
            line = linecache.getline(self.gensfile, idA).rstrip().split(' ')
            matresult.append(np.array(line[1:], dtype=np.int8))
        return(np.vstack(matresult))

#g = GensCache("C:/Users/mhindle/Documents/psuk_1_00", header=True)
#print(g.getMatrix(["3653120310902","3656123492012","3659126571902","3662126631109"]))
#print(g.getMatrix(["3653120310902","3656123492012","3659126571902","3662126631109"]).shape)


