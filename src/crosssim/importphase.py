'''
Created on Mar 13, 2021

@author: mhindle
'''
import csv, gzip
import numpy as np
from typing import Dict,List,Union
from numpy.typing import ArrayLike
import vcf
from progress.bar import Bar

class Phase():
    
    def __init__(self,tag:str,phap:ArrayLike, mhap:ArrayLike, chromosome:int, snpids:List[str]):
        self.tag = tag
        self.phap = phap
        self.mhap = mhap
        self.snpids = snpids
        self.chromosome = chromosome

def openfile(filename:str, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode) 
    else:
        return open(filename, mode)
        
def importAlphaImpute2(file:str, snpids:List[str], snpid2chr:Dict[str,int]) -> Dict[str,Dict[int,Phase]]: 
    '''
    file format .haplotypes file: no header two rows per animal. The first one is the maternal haplotype and the second is the paternal.
    '''
    haplotypes: Dict[str,Dict[int,Phase]] = {}
    #                tag,     chromosome
    chromosomeIndex = np.array([snpid2chr[x] for x in snpids])
    chromosomes = set(chromosomeIndex)
    snpids_f: ArrayLike = np.array(snpids)
    
    with openfile(file, mode='rt') as file_handle:
        lines = csv.reader(file_handle, delimiter=' ',quoting=csv.QUOTE_NONE)
        lastline: Union[List[str],List[None]] = [None]
        bar = Bar("Importing AlphaImpute2 file %s" % file, suffix='%(percent)d%%')
        for line in bar.iter(lines):
            tag = line[0]
            if lastline[0] == tag:
                #second entry
                mhap = np.array(lastline[1:]).astype(int)
                phap = np.array(line[1:]).astype(int)
                
                if tag not in haplotypes:
                    haplotypes[tag] = {} 
                
                for chromosome in chromosomes:
                    phap_chr = phap[chromosomeIndex == chromosome]
                    mhap_chr = mhap[chromosomeIndex == chromosome]
                    snpids_chr: ArrayLike = snpids_f[chromosomeIndex == chromosome]
                    haplotypes[tag][chromosome] = Phase(tag, phap=phap_chr, mhap=mhap_chr, chromosome=chromosome,snpids=snpids_chr)
            else:
                lastline = line
    return(haplotypes)

def importShapeit4(file:str) -> Dict[str,Phase]: 
    '''
    format is a phased vcf, so one SNP per row, animals in columns and variants are phased as per vcf spec (e.g 0|0 —> paternal and maternal  alleles)
    '''
    haplotypes: Dict[str,Phase] = {}
    print(file)
    with openfile(file, mode='rt') as file_handle:
        reader = vcf.Reader(file_handle)
        n_genotypes = len([i for i, record in enumerate(reader)]) # slow but better faster to do this and preinit np arrays
        print("%s genotypes in vcf" % n_genotypes)
        print("%s samples in vcf" % len(reader.samples))
        
        for sample in reader.samples :
            phap = np.full(n_genotypes, -9, dtype=int)
            mhap = np.full(n_genotypes, -9, dtype=int)
            tag = sample.split("_")[0]
            haplotypes[tag] = Phase(tag, phap=phap, mhap=mhap, snpids=[])
        reader = vcf.Reader(file_handle)
        for i, record in enumerate(reader):
            for sample in record.samples:
                tag = sample.sample.split("_")[0]
                pat_h, mat_h = sample['GT'].split("|")
                sample_hap = haplotypes[tag]
                sample_hap.phap[i] = int(pat_h)
                sample_hap.mhap[i] = int(mat_h)
                #print("%s %s" % (int(pat_h), sample_hap.phap[i]))
    
    return(haplotypes)
#
# #haplotypes = importShapeit4("/gensys/etarsani/gga2phasedallsim.vcf")
#
# #with open("/gensys/etarsani/gga2phasedallsim.haplotypes", mode='wt') as filehandle:
# ##    for tag,haplotype in haplotypes.items():
 # #       filehandle.write("%s %s\n" % (tag," ".join([str(x) for x in haplotype.phap])))
 # #       filehandle.write("%s %s\n" % (tag," ".join([str(x) for x in haplotype.mhap])))
        # #print("%s %s\n" % (tag," ".join([str(x) for x in haplotype.phap])))
        # #print("%s %s\n" % (tag," ".join([str(x) for x in haplotype.mhap])))
        #
# with openfile("/gensys/mhindle/quick_sim/poc3/snpIds.txt", mode='rt') as file_handle:
    # snpIds = [x.strip() for x in file_handle]
    #
# with openfile("/mnt/md0/mhindle/gensys/projects/GWAS/wooden/wooden_mapped.map") as file_handle:
    # snpdetails = {bla[1]:bla for bla in [x.strip().split("\t") for x in file_handle]}
    #
# print(len(snpdetails))
# print(len(snpIds))
# #print(len([x[1] for x in snpdetails.values() if x[1] in snpIds])) # check if we have all snp details
#
# chr2snps = [x for x in [snpdetails[x] for x in snpIds] if x[0] == "2"]
#
#
# print(chr2snps)
# haplotypes = importAlphaImpute2("/gensys/etarsani/gga2phasedallsim.haplotypes", snpids = snpIds)
# for tag,haplotype in haplotypes.items():
    # print("%s %s\n" % (tag," ".join([str(x) for x in haplotype.phap])))
    # print("%s %s\n" % (tag," ".join([str(x) for x in haplotype.mhap])))
    #


