
from dataclasses import dataclass,field
from typing import List, Optional    
from . import RAN_GEN, Genotype, np

@dataclass
class Chrom:
    name : str
    morgans: float = 0
    nvars : int = 0
    snpids: List[float] = field(default_factory=list)
    cm_pos: Optional[List[float]] = field(default_factory=list) or None
    
    def add_variant(self,snpid=None,cm_pos=None):
        self.nvars += 1
        if snpid:
            self.snpids.append(snpid) 
        if cm_pos:
            self.cm_pos.append(cm_pos) 
    
    def finalise_chrom_configuration(self):
        if self.cm_pos:
            self.cm_pos = np.array(self.cm_pos)
            self.nvars = len(self.cm_pos)

            if not self.morgans:
                self.morgans = max(self.cm_pos)/100

            ## normalise the distances to probabilities [NOTE probably exclude the first and last SNP]
            self.cm_to_p()
        else:
            self.cm_pos = None

            
    def cm_to_p(self,skip_first_last=True):
        ##check https://numpy.org/doc/stable/reference/generated/numpy.diff.html
        self.cm_pos = np.diff(self.cm_pos,prepend=0)/(self.morgans*100)
        if skip_first_last:
            self.cm_pos = self.cm_pos[1:-1]/self.cm_pos[1:-1].sum()
        else:
            self.cm_pos /= self.cm_pos.sum()

    def p_to_cm(self):
        self.cm_pos = np.cumsum(self.cm_pos*self.morgans*100)
        


class Genome:
    SWITCH = {0:1,1:0}
    
    def __init__(self,rs=None):
        self.chroms = {}
        self.rs = rs if rs else RAN_GEN
        
    def add_chrom(self,chrom_name):
        if chrom_name not in self.chroms:
            #print(f"will now add chromsome {chrom_name}")
            self.chroms[chrom_name] = Chrom(chrom_name)

    def add_variant(self,chrom_name,snpid=None,cm_pos=None):
        self.add_chrom(chrom_name)
        self.chroms[chrom_name].add_variant(snpid,cm_pos)
    
    def __repr__(self):
        info  = []
        for c in self.chroms:
                info.append(f"- Chromosome {c} contains {self.chroms[c].nvars} variants and is {self.chroms[c].morgans} MAP-UNITS long")
        return "\n".join(info)



    ##<Recombination> 
    def recomb_events(self,chrom_name,obligatory=1):
        if chrom_name in self.chroms:
            r = self.rs.poisson(self.chroms[chrom_name].morgans)
            return r if r >= 1 else obligatory
        return None
    
    def place_recomb(self,chrom,n_recomb,n_markers):
        if (n_markers > 1):        
            crossovers = self.rs.choice(np.arange(1,n_markers-1),n_recomb,p=chrom.cm_pos,replace=False)  ##avoid first and last markers to have clear recomb
            crossovers.sort()
            return crossovers.astype(int)
        return []
    
    def get_gamete(self,genotype):
        gamete = {c:[] for c in genotype.iterate_chroms()}
        crossovers = {c:[[],[]] for c in genotype.iterate_chroms()}
        
        for c in genotype.iterate_chroms():
            n_markers = self.chroms[c].nvars
            gamete[c] = np.empty(n_markers,dtype=int)
            
            ## determine recombination paramters for current chromosome c
            n_recomb = self.recomb_events(c,obligatory=1)
            breaks = self.place_recomb(self.chroms[c],n_recomb,n_markers)
            st_strand = self.rs.integers(0,2)
            
            ## Collate the gamete and record break points
            gamete[c][:breaks[0]] = genotype[c][st_strand][:breaks[0]]
            last_break = breaks[0]
            for i,b in enumerate(breaks[1:]):
                new_strand = self.SWITCH[(st_strand+i)%2]
                gamete[c][last_break:b] = genotype[c][new_strand][last_break:b]
                last_break = b
            
            ## Glue the last haplotype block after the last cross-over
            if len(breaks)%2 == 0:
                last_strand = st_strand
            else:
                last_strand = self.SWITCH[st_strand]
            gamete[c][last_break:] = genotype[c][last_strand][last_break:]
            crossovers[c] = [[st_strand,last_strand],breaks]
            
        return gamete,crossovers
    
    ##<Mutation>
    def mutate_gamete(self,gamete):
        """would switch alleles ONLY, not introducing new mutations"""
        pass
    ##</Mutation>
    
    def get_zygote(self,pat_gam,mat_gam):
        """BEWARE: No checks for dimension compatability """
        #genotype = {c:{0:[],1:[]} for c in self.chroms.keys()}
        genotype = Genotype(self.chroms)
        
        for c in self.chroms.keys():
            genotype[c][genotype.paternal_strand] = pat_gam[c]
            genotype[c][genotype.maternal_strand] = mat_gam[c]
        return genotype
