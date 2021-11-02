
from collections import UserDict
from . import RAN_GEN, np


class Genotype(UserDict):
    
    def __init__(self,chroms,rs=None,paternal_strand=0,maternal_strand=1):
        self.data = {}
        #self.strands = {''}
        for c in chroms:
            self.data[c] = {0:[],1:[]}
        self.rs = rs if rs else RAN_GEN
        self.paternal_strand = paternal_strand
        self.maternal_strand = maternal_strand


    def add_haplo_toStrand(self,chrom,strand,haplo_gen,mv=9, random_assign_missing=False):
        if strand in [self.paternal_strand,self.maternal_strand]:
            if random_assign_missing:
                self.data[chrom][strand] = np.where(haplo_gen==mv, self.rs.integers(0,2), haplo_gen)
            else:
                self.data[chrom][strand] = haplo_gen
    def iterate_chroms(self,specific_chroms=[]):
        chroms = self.data.keys()
        if specific_chroms:
            chroms = (c for c in specific_chroms if self.data.get(c,None))
        
        return chroms

    def as_haplo(self,strand):
        haplo,nsnps = {},0
        for c in self.iterate_chroms():
            haplo[c] =  self.data[c][strand]
            nsnps += len(haplo[c])
        return haplo,nsnps
    

    def as_snps(self):
        gen,nsnps = {},0
        for c in self.iterate_chroms():
            gen[c] =  self.data[c][0]+self.data[c][1]
            nsnps += len(gen[c])
        return gen,nsnps
    
    def vectorise_snp_gen(self,snp_gen,N):
        vec = np.empty(shape=N,dtype=int)
        st = 0
        for c in snp_gen.keys():
            vec[st:st+len(snp_gen[c])] = snp_gen[c]
            st += len(snp_gen[c])
        return vec
         
    def add_mv(self,gen_vec,prob=0.02,mv_code=9):
        n_events = int(len(gen_vec)*prob)
        mv_locs= self.rs.choice(len(gen_vec),n_events)
        gen_vec[mv_locs] = mv_code
        return gen_vec
    
    def add_error(self,gen_vec,prob):
        ## switch genotype. For example, if orginal gen was 0, then depending on x (being 0 or >0) return 1 or 2
        err_func = {0: lambda x:1 if x else 2, 1:lambda x:2 if x else 0, 2:lambda x:0 if x else 1} 
        
        n_events = int(len(gen_vec)*prob)
        err_locs = self.rs.choice(len(gen_vec),n_events)
        err_probs = self.rs.integers(0,2,size=len(err_locs))

        for idx,prob in zip(err_locs,err_probs):
            try:
                gen_vec[idx] = err_func[gen_vec[idx]](prob)
            except KeyError:
                pass
        return gen_vec
