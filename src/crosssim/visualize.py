'''
Created on Apr 22, 2021

@author: mhindle
'''
import math
import pickle
import scipy.sparse
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Dict, Tuple, Set
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
import numpy as np
import os
from matplotlib.lines import Line2D
from scipy.ndimage.filters import uniform_filter1d
import gzip
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from _asyncio import Future
import concurrent.futures
import itertools
import matplotlib.pyplot as plt
from collections import Counter
import psutil
from matplotlib.colors import LogNorm
import cloudpickle
from matplotlib import rc
import seaborn as sns
import mgzip
from itertools import chain
import gfutils

rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})
rc('mathtext',**{'default':'regular'})

def roundup(x, d):
    return int(math.ceil(x / d)) * d
                
def calcIndivMatrix(genotypelength, chromosome, kid, ni, in_dir, out_dir, returnmatrix=True, replaceall=False, threads=2):
    picklefile = '%s/sim_crossovers/pickle/chr%s/sim_ambigregions_%s.pickle.gz' % (in_dir,chromosome,kid)
    sims:List[Dict[int, Dict[int, List]]] = None #lazy load
    
    if replaceall or (in_dir != None and not Path('%s/sim_crossovers/npz/%s_%s_chr%s_crossprobs_pat.npz' % (in_dir,ni+1,kid,chromosome)).is_file()):
        if sims is None:
            sims = readPickle(picklefile, threads=threads)
        crossregion_pat = [sim[0] if sim is not None else None for sim in sims]
        crossprobs_pat = lil_matrix((genotypelength,len(sims)),dtype=np.float32) # we don't need 64bit precision
        for i, sim in enumerate(crossregion_pat):
            if sim is not None:
                pat_start, pat_length = sim
                for xstart, xend in zip(pat_start, (pat_start+pat_length)):
                    crossprobs_pat[xstart:xend,i] = 1./(float(xend-xstart)+1)
            else:
                crossprobs_pat[:,i] = 0
        scipy.sparse.save_npz('%s/sim_crossovers/npz/%s_%s_chr%s_crossprobs_pat.npz' % (out_dir,ni+1,kid,chromosome), crossprobs_pat.tocoo()) 
    elif returnmatrix:
        crossprobs_pat = lil_matrix(scipy.sparse.load_npz('%s/sim_crossovers/npz/%s_%s_chr%s_crossprobs_pat.npz' % (in_dir,ni+1,kid,chromosome)))
    
    if replaceall or (in_dir != None and not Path('%s/sim_crossovers/npz/%s_%s_chr%s_crossprobs_mat.npz' % (in_dir,ni+1,kid,chromosome)).is_file()):
        if sims is None:
            sims = readPickle(picklefile, threads=threads)
        crossregion_mat = [sim[1] if sim is not None else None for sim in sims]
        crossprobs_mat = lil_matrix((genotypelength,len(sims)),dtype=np.float32) # we don't need 64bit precision
        for i, sim in enumerate(crossregion_mat):
            if sim is not None:
                mat_start, mat_length  = sim
                for xstart, xend in zip(mat_start, (mat_start+mat_length)):
                    crossprobs_mat[xstart:xend,i] = 1./(float(xend-xstart)+1)
            else:
                crossprobs_mat[:,i] = 0
        scipy.sparse.save_npz('%s/sim_crossovers/npz/%s_%s_chr%s_crossprobs_mat.npz' % (out_dir,ni+1,kid,chromosome), crossprobs_mat.tocoo()) 
    elif returnmatrix:
        crossprobs_mat = lil_matrix(scipy.sparse.load_npz('%s/sim_crossovers/npz/%s_%s_chr%s_crossprobs_mat.npz' % (in_dir,ni+1,kid,chromosome)))

    if returnmatrix:
        actual:List[Dict[int, Dict[int, List]]] = readPickle('%s/sim_crossovers/pickle/chr%s/sim_actual_%s.pickle.gz' % (in_dir,chromosome,kid), threads=threads)
        crossactual_pat = [x[0] if x is not None else None for x in actual]
        crossactual_mat = [x[1] if x is not None else None for x in actual]
        
        if sims is None:
            sims = readPickle(picklefile, threads=threads)
            crossregion_pat = [sim[0] if sim is not None else None for sim in sims]
            crossregion_mat = [sim[1] if sim is not None else None for sim in sims]
        
        return(crossprobs_pat, crossprobs_mat, 
           crossactual_pat, crossactual_mat,
           crossregion_pat, crossregion_mat)

def loadRegionsStats(file, crossactual_pat, crossactual_mat, threads=2):
                        sims = readPickle(file, threads=threads)
                        crossregion_pat = [sim[0] for sim in sims]
                        crossregion_mat = [sim[1] for sim in sims]
                        statL = gfutils.Statistics()
                        statL.addCrossoverStatistics(crossactual_pat, crossregion_pat)
                        statL.addCrossoverStatistics(crossactual_mat, crossregion_mat)
                        return(crossregion_pat, crossregion_mat, statL)

def readPickle(file:str, threads=12):
    with mgzip.open(file, "rb", thread=threads) as picklef:
        return(pickle.load(picklef))

class Visualize(object):
    
    _BASEPAIR = "bp"
    _CHROMOSOME = "chr"
    
    def __init__(self, 
                    snpdetails: Dict[str, Dict[str, str]], # keys in nested "chr", "id", "cm","bp"
                    snpIds: List[str],
                    genotypelengths: Dict[int,int]):
        self.snpdetails = snpdetails
        self.snpIds = snpIds
        self.genotypelengths = genotypelengths
    # body of the constructor

    def realxoverStatistics(self,xovers,countxovers,chromosomes,fout):
    
        #[chrom] = {0:patcregion, 1:matcregion}
        #print(xovers)
        print(np.array(list(countxovers.values())).shape)
        
        generation = [str(x)[0:3] for x in countxovers.keys()]
        
        xoverCounts = pd.DataFrame(np.array(list(countxovers.values())), columns=chromosomes, index=countxovers.keys())
        xoverCounts_long = pd.melt(xoverCounts.reset_index(), id_vars='index')
        print(xoverCounts_long.head(5))
        xoverCounts_long.columns = ['Kid', 'Chromosome','xovers']
        xoverCounts_long["Generation"] = [str(x)[0:2] for x in xoverCounts_long["Kid"] ]
        xoverCounts_long["log10_xovers"] = [math.log(x,10) if x > 0 else 0 for x in xoverCounts_long["xovers"] ]
        
        with PdfPages(fout) as pdf:
            fig = plt.figure(figsize=(12,24), dpi= 600, facecolor='w', edgecolor='k')
            ax = sns.boxplot(y="Chromosome", x="log10_xovers",data=xoverCounts_long, orient="h", hue="Generation")
            pdf.savefig()
            plt.close()
            
            fig = plt.figure(figsize=(48,24), dpi= 600, facecolor='w', edgecolor='k')
            ax = sns.boxplot(y="Chromosome", x="xovers",data=xoverCounts_long, orient="h", hue="Generation")
            pdf.savefig()
            plt.close()
            
            for chromosome in chromosomes:
                print("chromosome %s" % chromosome)
                lengthvspos_all  = self.ambiguityHeatMatrix(self.genotypelengths[chromosome],
                                                             [xover[chromosome][0] for kid, xover in xovers.items()],
                                                             [xover[chromosome][1] for kid, xover in xovers.items()])
                
                smooth_all = uniform_filter1d(lengthvspos_all.transpose().sum(axis=0).tolist(), size=20)
                smooth_less50 = uniform_filter1d(lengthvspos_all[:,0:50].transpose().sum(axis=0).tolist(), size=20)
                
                print("n pos %s max length %s " % lengthvspos_all.shape)
                
                fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
                pos = plt.imshow(lengthvspos_all.transpose(), 
                                     interpolation='none', cmap=plt.cm.get_cmap('winter').reversed(), origin='lower', aspect=2, norm=LogNorm())
                plt.plot(range(0, len(smooth_all)),
                         gfutils.scale_list(smooth_all,0,lengthvspos_all.shape[1],0, max(smooth_all)), color="blue", lw=0.5)
                plt.plot(range(0, len(smooth_less50)),
                         gfutils.scale_list(smooth_less50,0,lengthvspos_all.shape[1],0, max(smooth_all)), color="blue", lw=0.5, linestyle='dashed')
                
                secaxy = plt.gca().secondary_yaxis('right', 
                                       functions=(lambda a: gfutils.scale_number(a,0,lengthvspos_all.shape[1], 0,max(smooth_all)), 
                                                  lambda a: gfutils.scale_number(a,0,max(smooth_all),0,lengthvspos_all.shape[1])))
                secaxy.set_ylabel('Density (50 bp window)')
                secaxy.set_color('blue')

                plt.xlabel('N genomic position (chr %s)' % chromosome)
                plt.ylabel('length of ambiguity region')
                plt.title('Chromosome %s \nlocation density by length of ambiguity x-over region' % chromosome)
                cb = fig.colorbar(pos)
                cb.set_label('log(N)')
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
    
    def populationStatistics(self, chromosomes: List[int], 
                              kids: List[str],
                              in_dir:str, 
                              out_dir:str, 
                              threads=int(mp.cpu_count()/2)):
        
        #ensure they are created but do not load
        print("pregenerate individual matrix probs")
        self.createProbMatrix(chromosomes, kids, in_dir,out_dir, threads=threads)
        print("done")
        os.makedirs("%s/sim_crossovers/population_sim" % (out_dir),exist_ok=True)
        process = psutil.Process(os.getpid())
        
        for chromosome in chromosomes:
            os.makedirs("%s/sim_crossovers/population_sim/chr%s" % (out_dir, chromosome),exist_ok=True)
            genotypelength = self.genotypelengths[chromosome]

            crossactual_mat_col = {}
            crossactual_pat_col = {}

            n_sims_pat_sum = 0
            n_sims_mat_sum = 0
            
            pat_exactfile = '%s/sim_crossovers/population_sim/chr%s/sim_exact_pat.pickle.gz' % (out_dir,chromosome)
            mat_exactfile = '%s/sim_crossovers/population_sim/chr%s/sim_exact_mat.pickle.gz' % (out_dir,chromosome)
            
            if not (os.path.exists(pat_exactfile) and os.path.isfile(pat_exactfile) and os.path.exists(mat_exactfile) and os.path.isfile(mat_exactfile)):
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    jobs = {executor.submit(readPickle, '%s/sim_crossovers/pickle/chr%s/sim_actual_%s.pickle.gz' % (in_dir,chromosome,kid)):(kid, chromosome) for kid in kids }
                    print("waiting on unpickle work for %s jobs on %s threads for chromosome %s...." % (len(jobs), threads, chromosome))
                    for ni, future in enumerate(concurrent.futures.as_completed(jobs) ):
                            kid, chromosome = jobs[future]
                            e = future.exception()
                            if e is not None:
                                raise e
                            actual = future.result()
                            crossactual_pat_col[kid] = [x[0] for x in actual]
                            crossactual_mat_col[kid] = [x[1] for x in actual]
                            del jobs[future]
                            del actual
                            del future
                            #print("done %s and %s left: memory usage %0.2f MB (%0.2f pc)" % (ni+1, len(jobs), process.memory_info().rss / 1024 ** 2, process.memory_percent()))
                print("Save to pickle %s %s" % (pat_exactfile, mat_exactfile))
                
                gfutils.dumpToPickle(pat_exactfile, crossactual_pat_col)
                gfutils.dumpToPickle(mat_exactfile, crossactual_mat_col)
                #print("actual x-cross: loaded %s simulations" % len(actual))
            else :
                print("Load from pickle %s %s" % (pat_exactfile, mat_exactfile))
                crossactual_pat_col = readPickle(pat_exactfile, threads=threads)
                crossactual_mat_col = readPickle(mat_exactfile, threads=threads)
                #print("done load: memory usage %0.2f MB (%0.2f pc)" % (process.memory_info().rss / 1024 ** 2, process.memory_percent()))
            
            pat_probsfile = '%s/sim_crossovers/population_sim/chr%s/sim_probs_pat.pickle.gz' % (out_dir,chromosome)
            mat_probsfile = '%s/sim_crossovers/population_sim/chr%s/sim_probs_mat.pickle.gz' % (out_dir,chromosome)
            if not (os.path.exists(pat_probsfile) and os.path.isfile(pat_probsfile) and os.path.exists(mat_probsfile) and os.path.isfile(mat_probsfile)):
                print("build population crossprobs matrix")
                crossprobs_mat_col = np.empty((genotypelength,0), dtype=np.float32)
                crossprobs_pat_col = np.empty((genotypelength,0), dtype=np.float32)
                for ni, kid in enumerate(kids) :
                    crossprobs_pat = scipy.sparse.load_npz('%s/sim_crossovers/npz/%s_%s_chr%s_crossprobs_pat.npz' % (out_dir,ni+1,kid,chromosome)) 
                    crossprobs_mat = scipy.sparse.load_npz('%s/sim_crossovers/npz/%s_%s_chr%s_crossprobs_mat.npz' % (out_dir,ni+1,kid,chromosome)) 
                    n_geno_pat, n_sims_pat = crossprobs_pat.shape
                    n_geno_mat, n_sims_mat = crossprobs_mat.shape
                    n_sims_pat_sum += n_sims_pat
                    n_sims_mat_sum += n_sims_mat
                    crossprobs_pat_col = np.hstack([crossprobs_pat_col, crossprobs_pat.sum(axis=1).tolist()])
                    crossprobs_mat_col = np.hstack([crossprobs_mat_col, crossprobs_mat.sum(axis=1).tolist()])
                gfutils.dumpToPickle(pat_probsfile, crossprobs_pat_col)
                gfutils.dumpToPickle(mat_probsfile, crossprobs_mat_col)
            else :
                print("Load from pickle %s %s" % (pat_probsfile, mat_probsfile))
                crossprobs_pat_col = readPickle(pat_probsfile, threads=threads)
                crossprobs_mat_col = readPickle(mat_probsfile, threads=threads)
                n_sims_pat_sum = sum([len(x) for x in crossactual_pat_col.values()])
                n_sims_mat_sum = sum([len(x) for x in crossactual_mat_col.values()])
                #print("nsims pat %s mat %s " %(n_sims_pat_sum, n_sims_pat_sum))
                #print("done load: memory usage %0.2f MB (%0.2f pc)" % (process.memory_info().rss / 1024 ** 2, process.memory_percent()))
            
            print("Write sim_paternal region txt")
            with mgzip.open('%s/sim_crossovers/population_sim/sim_paternal_chr%s.txt.gz' % (out_dir,chromosome), "wt", thread=threads) as fout:
                fout.write("#\t%s\n" % ("\t".join(self.snpIds)))
                fout.write("#\t%s\n" % ("\t".join(map(str, [self.snpdetails[x][self._CHROMOSOME] for x in self.snpIds]))))
                fout.write("#\t%s\n" % ("\t".join([self.snpdetails[x][self._BASEPAIR] for x in self.snpIds])))
                for kid, row in zip(kids, crossprobs_pat_col):
                    fout.write("%s\t%s\n" % (kid, "\t".join(map(str, row))))
            
            print("Write sim_maternal region txt")
            with mgzip.open('%s/sim_crossovers/population_sim/sim_maternal_chr%s.txt.gz' % (out_dir,chromosome), "wt", thread=threads) as fout:
                fout.write("#\t%s\n" % ("\t".join(self.snpIds)))
                fout.write("#\t%s\n" % ("\t".join(map(str, [self.snpdetails[x][self._CHROMOSOME] for x in self.snpIds]))))
                fout.write("#\t%s\n" % ("\t".join([self.snpdetails[x][self._BASEPAIR] for x in self.snpIds])))
                for kid, row in zip(kids, crossprobs_mat_col):
                    fout.write("%s\t%s\n" % (kid, "\t".join(map(str, row))))
            
            with PdfPages('%s/sim_crossovers/population_sim/chr%s_plotsims.pdf' %  (out_dir,chromosome) ) as pdf:
                fig=plt.figure(figsize=(12,4), dpi= 200, facecolor='w', edgecolor='k')
                plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                         [x/n_sims_pat_sum for x in crossprobs_pat_col.sum(axis=1).tolist()], color="blue", lw=1)
                plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                         [x/n_sims_mat_sum for x in crossprobs_mat_col.sum(axis=1).tolist()], color="purple", lw=1)
                plt.xlabel('position genome (pos Mbp)')
                plt.ylabel('crossover p̂ (sum p at position/n simulations)')
                plt.title('Crossover p̂ at genomic position\nin maternal and paternal haplotype')
                plt.legend([
                    Line2D([0], [0], color="blue", lw=1),
                    Line2D([0], [0], color="purple", lw=1)] , ["paternal","maternal"])
                pdf.savefig()
                
                smooth_pat = uniform_filter1d([x/n_sims_pat_sum for x in crossprobs_pat_col.sum(axis=1).tolist()], size=50)
                smooth_mat = uniform_filter1d([x/n_sims_mat_sum for x in crossprobs_mat_col.sum(axis=1).tolist()], size=50)
                
                fig=plt.figure(figsize=(12,4), dpi= 200, facecolor='w', edgecolor='k')
                plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                         smooth_pat, color="blue", lw=1)
                plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                         smooth_mat, color="purple", lw=1)
                plt.xlabel('position genome (pos Mbp)')
                plt.ylabel('50 snp smoothed\np̂ (sum p at position/n simulations)')
                plt.title('Crossover p̂ at genomic position\nin maternal and paternal haplotype')
                plt.legend([
                    Line2D([0], [0], color="blue", lw=1),
                    Line2D([0], [0], color="purple", lw=1)] , ["paternal","maternal"])
                pdf.savefig()
                
                # fig=plt.figure(figsize=(12,4), dpi= 200, facecolor='w', edgecolor='k')
                # for row in crossprobs_pat_col.transpose():
                    # smooth_pat_each = uniform_filter1d([x/n_sims_pat_sum for x in row], size=50)
                    # plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                             # smooth_pat_each, lw=0.33)
                # plt.xlabel('position genome (pos Mbp)')
                # plt.ylabel('50 snp smoothed\np̂ (sum p at position/n simulations)')
                # plt.title('Crossover p̂ at genomic position for paternal haplotypes')
                # pdf.savefig()
                # plt.close(fig)
                #
                # fig=plt.figure(figsize=(12,4), dpi= 200, facecolor='w', edgecolor='k')
                # for row in crossprobs_pat_col.transpose():
                    # smooth_pat_each = uniform_filter1d([x/n_sims_pat_sum for x in row], size=50)
                    # plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                             # smooth_pat_each, lw=0.33)
                # plt.xlabel('position genome (pos Mbp)')
                # plt.ylabel('50 snp smoothed\np̂ (sum p at position/n simulations)')
                # plt.title('Crossover p̂ at genomic position for maternal haplotypes')
                # pdf.savefig()
                # plt.close(fig)
                #
                pca = PCA(n_components=2)
                components = pca.fit_transform(
                    np.concatenate((crossprobs_pat_col.transpose(), crossprobs_mat_col.transpose()), axis=0))
                
                principalDf = pd.DataFrame(data = components, columns = ['principal component 1', 'principal component 2'])
                principalDf["ID"] = list(kids)+list(kids)
                principalDf["line"] = ["paternal"]*len(kids)+["maternal"]*len(kids)

                fig = plt.figure(figsize = (8,8), dpi= 200, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(1,1,1) 
                ax.set_xlabel('Principal Component 1', fontsize = 15)
                ax.set_ylabel('Principal Component 2', fontsize = 15)
                ax.set_title('2 component PCA', fontsize = 20)
                lines = ['paternal', 'maternal']
                colors = ['blue', 'purple']
                for line, color in zip(lines,colors):
                    indicesToKeep = principalDf['line'] == line
                    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                               , principalDf.loc[indicesToKeep, 'principal component 2']
                               , c = color
                               , s = 10)
                ax.legend(lines)
                ax.grid()
                pdf.savefig()
                plt.close(fig)
                
                print("plotting accuracy statistics pop")
                lengthvspos_all_file = '%s/sim_crossovers/population_sim/chr%s/densitylengthposition.pickle.gz' %  (out_dir,chromosome)
                perfomstats_file = '%s/sim_crossovers/population_sim/chr%s/performancestats.pickle.gz' %  (out_dir,chromosome)
                
                recalcLenStats = False
                if os.path.exists(lengthvspos_all_file) and os.path.isfile(lengthvspos_all_file):
                    lengthvspos_all = readPickle(lengthvspos_all_file)
                    print("loaded by pickle %s by %s matrix for length stats" % lengthvspos_all.shape)
                else :
                    recalcLenStats = True
                    lengthvspos_all = []
                
                performStats = False
                if os.path.exists(perfomstats_file) and os.path.isfile(perfomstats_file):
                    stats = readPickle(perfomstats_file)
                    print("loaded by pickle performance stats")
                else :
                    performStats = True
                    stats = gfutils.Statistics()
                
                if performStats or recalcLenStats :
                    with ProcessPoolExecutor(max_workers=threads) as executor:
                        jobs = {executor.submit(loadRegionsStats, '%s/sim_crossovers/pickle/chr%s/sim_ambigregions_%s.pickle.gz' % (in_dir,chromosome,kid),crossactual_pat_col[kid],crossactual_mat_col[kid], threads=threads):(kid, chromosome) for kid in kids }
                        for ni, future in enumerate(concurrent.futures.as_completed(jobs)) :
                            kid, chromosome = jobs[future]
                            e = future.exception()
                            if e is not None:
                                print("Exception %s" % e)
                                raise Exception(e)
                            crossregion_pat, crossregion_mat, stats_indiv = future.result()
    
                            #print("processing %s of %s: memory usage %0.2f MB (%0.2f pc)"  % (ni+1, len(kids), process.memory_info().rss / 1024 ** 2, process.memory_percent()))
                            del jobs[future]
                            del future
                            
                            if recalcLenStats :
                                print("calc length vs pos")
                                lengthvspos = self.ambiguityHeatMatrix(crossprobs_pat.shape[0],
                                         crossregion_pat,crossregion_mat)
                                if len(lengthvspos_all) == 0:
                                    lengthvspos_all = lengthvspos
                                else:
                                    lengthvspos_all = gfutils.addPaddedArrays(lengthvspos_all,lengthvspos)
                            
                            if performStats :
                                print("add crossover statistics for %s " % kid)
                                stats.mergeStatistics(stats_indiv)
                            #print("%s of %s: memory usage %0.2f MB (%0.2f pc)" % (ni+1, len(kids), process.memory_info().rss / 1024 ** 2, process.memory_percent()))
                                
                    if recalcLenStats :    
                        print("write pickle region length denstiy stats %s " % lengthvspos_all_file)
                        gfutils.dumpToPickle(lengthvspos_all_file, lengthvspos_all, threads=3)
                        
                    if performStats :   
                        print("write pickle performance stats %s " % perfomstats_file)
                        gfutils.dumpToPickle(perfomstats_file, stats, replace=True, threads=3)
                
                print("done length stats gen")
            
                fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
                pos = plt.imshow(lengthvspos_all.astype(float).transpose(), 
                                 interpolation='none', cmap=plt.cm.get_cmap('winter').reversed(), origin='lower', aspect=3, norm=LogNorm())
                plt.xlabel('N genomic position')
                plt.ylabel('length of ambiguity region')
                plt.title('Location density by length of ambiguity x-over region')
                cb = fig.colorbar(pos)
                cb.set_label('log(N)')
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
                
                fig =plt.figure(figsize=(6,4), dpi= 300, facecolor='w', edgecolor='k')
                #print(stats.n_actual)
                maxvalue = max(stats.n_actual+stats.n_detected)
                #hb = plt.hexbin(n_actual, n_detected, bins='log', cmap='cool', mincnt=1, extent=[0,maxvalue,0,maxvalue])
                
                sumintersect = np.zeros((maxvalue+1, maxvalue+1), dtype=int)
                for act, infer in zip(stats.n_actual,stats.n_detected) :
                    sumintersect[infer][act] += 1
                
                pos = plt.imshow(sumintersect, interpolation='none', cmap=plt.cm.get_cmap('Reds_r').reversed(), origin='lower')
                plt.yticks(np.arange(0, maxvalue+1, 1))
                plt.xticks(np.arange(0, maxvalue+1, 1))
                plt.xlabel('n actual xovers')
                plt.ylabel('n detected xovers')
                plt.title('actual vs predicted x-overs')
                cb = fig.colorbar(pos)
                cb.set_label('N')
                linear = [x for x in range(0, maxvalue)]
                plt.plot(linear, linear)
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
                
                fig =plt.figure(figsize=(6,4), dpi= 300, facecolor='w', edgecolor='k')
                pos = plt.imshow(sumintersect, interpolation='none', cmap=plt.cm.get_cmap('Reds_r').reversed(), origin='lower', norm=LogNorm())
                plt.yticks(np.arange(0, maxvalue+1, 1))
                plt.xticks(np.arange(0, maxvalue+1, 1))
                plt.xlabel('n actual xovers')
                plt.ylabel('n detected xovers')
                plt.title('actual vs predicted x-overs')
                cb = fig.colorbar(pos)
                cb.set_label('log(N)')
                linear = [x for x in range(0, maxvalue)]
                plt.plot(linear, linear)
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
                
                fig =  plt.figure(figsize=(6,4), dpi= 300)
                counter: Dict[int, int] = dict(sorted(Counter(stats.n_actual2predicted).items()))
                plt.bar(counter.keys(), counter.values(), edgecolor="black")
                plt.yticks(np.arange(0, roundup(max(counter.values()), 1000)+1, roundup(roundup(max(counter.values()), 1000)/5., 5)))
                plt.xticks([])  
                #plt.xticks(np.arange(0, math.ceil(max(map(int, counter.keys())))+1, 1))
                plt.title('Actual x-overs\nwith N overlapping predicted regions')
                plt.xlabel('N ACTUAL x-overs')
                plt.ylabel('N overlapping PREDICTED regions')
                sns.despine(top=True, right=True, left=False, bottom=False)
                plt.table(cellText=[list(counter.values())],
                      rowLabels=["xovers"],
                      colLabels=list(counter.keys()),
                      loc='bottom',colLoc='center', rowLoc='center')
                pdf.savefig()
                plt.close(fig)
                
                fig =  plt.figure(figsize=(6,4), dpi= 300)
                counter: Dict[int, int] = dict(sorted(Counter(stats.n_predicte2actual).items()))
                plt.bar(counter.keys(), counter.values(), edgecolor="black")
                plt.yticks(np.arange(0, roundup(max(counter.values()), 1000)+1, roundup(roundup(max(counter.values()), 1000)/5., 5)))
                plt.xticks([])  
                #plt.xticks(np.arange(0, math.ceil(max(map(int, counter.keys())))+1, 1))
                plt.title('Predicted regions\nwith N overlapping actual x-overs')
                plt.xlabel('N PREDICTED regions overlaps')
                plt.ylabel('N overlapping ACTUAL x-over')
                sns.despine(top=True, right=True, left=False, bottom=False)
                plt.table(cellText=[list(counter.values())],
                      rowLabels=["xovers"],
                      colLabels=list(counter.keys()),
                      loc='bottom',colLoc='center', rowLoc='center')
                pdf.savefig()
                plt.close(fig)
                
    def createProbMatrix(self,
                              chromosomes: List[int], 
                              kids: List[str], 
                              in_dir:str,
                              out_dir:str, 
                              threads=int(mp.cpu_count()/2)):       
        os.makedirs("%s/sim_crossovers/npz" % (out_dir),exist_ok=True)  
        process = psutil.Process(os.getpid())
        jobs = {}
        with ProcessPoolExecutor(max_workers=threads) as executor:
            for ni, kid in enumerate(kids) :
                for chromosome in chromosomes:
                    genotypelength = self.genotypelengths[chromosome]
                    jobs[executor.submit(calcIndivMatrix, genotypelength, chromosome, kid, ni, in_dir, out_dir, returnmatrix=False)] = (kid, chromosome)
        
        for ni, future in enumerate(concurrent.futures.as_completed(jobs) ):
            kid, chromosome = jobs[future]
            e = future.exception()
            if e is not None:
                raise e
            del jobs[future]
            del future
    
    def individualsStatistics(self, 
                              chromosomes: List[int], 
                              kids: List[str], 
                              in_dir:str,
                              out_dir:str, 
                              threads=int(mp.cpu_count()/2)) :
        '''
        '''
        os.makedirs("%s/sim_crossovers/npz" % (out_dir),exist_ok=True)
            
        process = psutil.Process(os.getpid())
        jobs = {}
        with ProcessPoolExecutor(max_workers=threads) as executor:
            for ni, kid in enumerate(kids) :
                for chromosome in chromosomes:
                    genotypelength = self.genotypelengths[chromosome]
                    jobs[executor.submit(calcIndivMatrix, genotypelength, chromosome, kid, ni, in_dir, out_dir, returnmatrix=True)] = (kid, chromosome)
            
            print("waiting on matrix work....")
            for ni, future in enumerate(concurrent.futures.as_completed(jobs) ):
                    kid, chromosome = jobs[future]
                    e = future.exception()
                    if e is not None:
                        raise e
                    crossprobs_pat, crossprobs_mat, crossactual_pat, crossactual_mat, crossregion_pat, crossregion_mat = future.result()
                    print("plotting for kid %s chromosome %s " % (kid,chromosome))
                    #no point plotting stats for chromosomes with no simulated crossovers
                    if len([x for x in crossactual_pat if x is not None]) > 0 and len([x for x in crossactual_mat if x is not None]) > 0 :
                        self.individualStatistics(chromosome, 
                                              crossprobs_pat, crossprobs_mat, 
                                              crossactual_pat,crossactual_mat,
                                              crossregion_pat,crossregion_mat,
                                              kid, out_dir)
                    del jobs[future]
                    del future
                    del crossprobs_pat
                    del crossprobs_mat
                    del crossactual_pat
                    del crossactual_mat
                    del crossregion_pat
                    del crossregion_mat
                    #print("%s done %s left: memory usage %0.2f MB (%0.2f pc)" % (ni+1, len(jobs), process.memory_info().rss / 1024 ** 2, process.memory_percent()))
    
    def ambiguityHeatMatrix(self,
                             genomiclength:int,
                             crossregion_pat_col,crossregion_mat_col, defaultupper = 500):
        '''
        this function is SO UGLY TODO: redo
        '''
        maxlen = defaultupper
        if len(crossregion_pat_col[0]) > 0:
            for start_r, len_r in crossregion_pat_col:  
                for l in len_r:
                    maxlen = max(maxlen, l)
        
        if len(crossregion_mat_col[0]) > 0:
            for start_r, len_r in crossregion_mat_col:  
                for l in len_r:
                    maxlen = max(maxlen, l)
        
        lengthvspos = np.zeros((genomiclength,maxlen+1), dtype=int) 
        
        #print("crossregion_pat_col count lengths")
        if len(crossregion_pat_col[0]) > 0:
            for (start_r, len_r) in crossregion_pat_col:  
                for s,l in zip(start_r, len_r):
                    for x in range(s, s+l):
                        lengthvspos[x,l]+=1
        
        #print("crossregion_mat_col count lengths")
        if len(crossregion_mat_col[0]) > 0:
            for (start_r, len_r) in crossregion_mat_col:  
                for s,l in zip(start_r, len_r):
                    for x in range(s, s+l):
                        lengthvspos[x,l]+=1
        
        return(lengthvspos)
    
    def individualStatistics(self, chromosome:int,
                             crossprobs_pat: lil_matrix,crossprobs_mat: lil_matrix,
                             crossactual_pat_col,crossactual_mat_col,
                             crossregion_pat_col,crossregion_mat_col,
                             kid : str,
                             out_dir:str) :
            '''
            '''
            os.makedirs("%s/sim_crossovers/plot_individuals/chr%s/" % (out_dir,chromosome), exist_ok=True)
            
            n_geno_pat, n_sims_pat = crossprobs_pat.shape
            n_geno_mat, n_sims_mat = crossprobs_mat.shape
            
            with PdfPages('%s/sim_crossovers/plot_individuals/chr%s/%s_plotsims.pdf' %  (out_dir,chromosome,kid) ) as pdf:
                
                position_actuals_pat: Dict[int, int] = Counter()
                for x in crossactual_pat_col:
                    position_actuals_pat.update(x)
                position_actuals_pat: Dict[int, int] = dict(sorted(position_actuals_pat.items()))  
                
                position_actuals_mat: Dict[int, int] = Counter()
                for x in crossactual_mat_col:
                    position_actuals_mat.update(x)
                position_actuals_mat: Dict[int, int] =  dict(sorted(position_actuals_mat.items())) 
                
                #print(list(position_actuals_mat.items())[1:10])
                
                smooth_pat_exact = uniform_filter1d([x/n_geno_pat for x in position_actuals_pat.values()], size=50)
                
                fig=plt.figure(figsize=(12,4), dpi= 300, facecolor='w', edgecolor='k')
                plt.plot(position_actuals_pat.keys(),
                         [x/n_geno_pat for x in position_actuals_pat.values()], color="blue", lw=0.3)
                plt.plot(position_actuals_pat.keys(),
                         smooth_pat_exact, '--', color="black", lw=1)
                plt.xlabel('marker (N)')
                plt.ylabel('p̂ of x-over (p̂ is snp exact)')
                plt.title('Crossover p̂ at genomic position paternal haplotype')
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close()
                
                smooth_mat_exact = uniform_filter1d([x/n_geno_mat for x in position_actuals_mat.values()], size=50)
                
                fig=plt.figure(figsize=(12,4), dpi= 300, facecolor='w', edgecolor='k')
                plt.plot(position_actuals_mat.keys(),
                         [x/n_geno_mat for x in position_actuals_mat.values()], color="purple", lw=0.3)
                plt.plot(position_actuals_mat.keys(),
                         smooth_mat_exact, '--', color="black", lw=1)
                plt.xlabel('marker (N)')
                plt.ylabel('p̂ of x-over (p̂ is snp exact)')
                plt.title('Crossover p̂ at genomic position maternal haplotype')
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close()
                
                
                fig=plt.figure(figsize=(12,4), dpi= 300, facecolor='w', edgecolor='k')
                plt.plot([x for x in range(0,n_geno_pat)],
                         [x[0]/n_sims_pat for x in crossprobs_pat.sum(axis=1).tolist()], color="blue", lw=0.3)
                plt.plot([x for x in range(0,n_geno_mat)],
                         [x[0]/n_sims_mat for x in crossprobs_mat.sum(axis=1).tolist()], color="purple", lw=0.3)

                plt.xlabel('marker (N)')
                plt.ylabel('p̂ of x-over (p̂ is distributed by length for ambiguous regions)')
                plt.title('Crossover prediction p̂ at genomic position\nin maternal and paternal haplotype')
                plt.legend([
                    Line2D([0], [0], color="blue", lw=1),
                    Line2D([0], [0], color="purple", lw=1)] , ["paternal","maternal"])
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close()
                
                
                fig=plt.figure(figsize=(12,4), dpi= 300, facecolor='w', edgecolor='k')
                plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                         [x[0]/n_sims_pat for x in crossprobs_pat.sum(axis=1).tolist()], color="blue", lw=0.3)
                plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                         [x[0]/n_sims_mat for x in crossprobs_mat.sum(axis=1).tolist()], color="purple", lw=0.3)
                plt.xlabel('position genome (pos Mbp)')
                plt.ylabel('crossover p̂ (sum p at position/n simulations)')
                plt.title('Crossover p̂ at genomic position\nin maternal and paternal haplotype')
                plt.legend([
                    Line2D([0], [0], color="blue", lw=1),
                    Line2D([0], [0], color="purple", lw=1)] , ["paternal","maternal"])
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
                
                smooth_pat = uniform_filter1d([x[0]/n_sims_pat for x in crossprobs_pat.sum(axis=1).tolist()], size=50)
                smooth_mat = uniform_filter1d([x[0]/n_sims_mat for x in crossprobs_mat.sum(axis=1).tolist()], size=50)
                
                fig=plt.figure(figsize=(12,4), dpi= 300, facecolor='w', edgecolor='k')
                plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                         smooth_pat, color="blue", lw=1)
                plt.plot([int(self.snpdetails[x][self._BASEPAIR])/1000000 for x in self.snpIds if int(self.snpdetails[x][self._CHROMOSOME]) == chromosome],
                         smooth_mat, color="purple", lw=1)
                plt.plot(position_actuals_pat.keys(),
                         [x/n_sims_pat for x in position_actuals_pat.values()], '--', color="blue", lw=0.5)
                plt.plot(position_actuals_mat.keys(),
                         [x/n_sims_pat for x in position_actuals_mat.values()], '--', color="purple", lw=0.5)
                plt.xlabel('position genome (pos Mbp)')
                plt.ylabel('p̂ (sum p at position/n simulations)')
                plt.title('50 snp smoothed crossover p̂ at genomic position\nin maternal and paternal haplotype')
                plt.legend([
                    Line2D([0], [0], color="blue", lw=1),
                    Line2D([0], [0], color="purple", lw=1),
                    Line2D([0], [0], linestyle='--', color="blue", lw=0.5),
                    Line2D([0], [0], linestyle='--', color="purple", lw=0.5)
                ] , ["predicted region paternal","predicted region maternal","exact x-over paternal","exact x-over maternal"])
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
                                
                #plt.hist2d(, ,)
                print("plotting accuracy statistics")

                stats = gfutils.Statistics()
                #paternal stats
                stats.addCrossoverStatistics(crossactual_pat_col, crossregion_pat_col)
                stats.addCrossoverStatistics(crossactual_mat_col, crossregion_mat_col)
                
                fig =  plt.figure(figsize=(6,4), dpi= 300)
                plt.bar(stats.length_overlap_actuals.keys(), stats.length_overlap_actuals.values(), edgecolor="black")
                plt.title('Distribution ambiguous region lengths for simulated x-overs')
                plt.xlabel('region length')
                plt.ylabel('simulated x-overs')
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()  
                plt.close(fig)
                
                fig =plt.figure(figsize=(6,4), dpi= 300, facecolor='w', edgecolor='k')
                maxvalue = max(stats.n_actual+stats.n_detected+[0])
                #hb = plt.hexbin(n_actual, n_detected, bins='log', cmap='cool', mincnt=1, extent=[0,maxvalue,0,maxvalue])
                
                sumintersect = np.zeros((maxvalue+1, maxvalue+1), dtype=int)
                for act, infer in zip(stats.n_actual,stats.n_detected) :
                    sumintersect[infer][act] += 1
                
                pos = plt.imshow(sumintersect, interpolation='none', cmap=plt.cm.get_cmap('Reds_r').reversed(), origin='upper')
                plt.yticks(np.arange(0, maxvalue+1, 1))
                plt.xticks(np.arange(0, maxvalue+1, 1))
                plt.xlabel('n actual xovers')
                plt.ylabel('n detected xovers')
                plt.title('actual vs predicted x-overs')
                cb = fig.colorbar(pos)
                cb.set_label('N')
                linear = [x for x in range(0, maxvalue)]
                plt.plot(linear, linear)
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
                
                fig =plt.figure(figsize=(6,4), dpi= 300, facecolor='w', edgecolor='k')
                pos = plt.imshow(sumintersect, interpolation='none', cmap=plt.cm.get_cmap('Reds_r').reversed(), origin='upper', norm=LogNorm())
                plt.yticks(np.arange(0, maxvalue+1, 1))
                plt.xticks(np.arange(0, maxvalue+1, 1))
                plt.xlabel('n actual xovers')
                plt.ylabel('n detected xovers')
                plt.title('actual vs predicted x-overs')
                cb = fig.colorbar(pos)
                cb.set_label('log(N)')
                linear = [x for x in range(0, maxvalue)]
                plt.plot(linear, linear)
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
                
                lengthvspos = self.ambiguityHeatMatrix(crossprobs_pat.shape[0],
                             crossregion_pat_col,crossregion_mat_col)
                
                #gfutils.dumpToPickle('%s/sim_crossovers/plot_individuals/chr%s/%s.pickle' %  (out_dir,chromosome,kid), lengthvspos)
                
                fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
                pos = plt.imshow(lengthvspos.astype(float).transpose(), #removed .toarray()
                                 interpolation='none', cmap=plt.cm.get_cmap('winter').reversed(), origin='lower', aspect=20, norm=LogNorm())
                plt.xlabel('N genomic position')
                plt.ylabel('length of ambiguity region')
                plt.title('Location density by length of ambiguity x-over region')
                cb = fig.colorbar(pos)
                cb.set_label('log(N)')
                sns.despine(top=True, right=True, left=False, bottom=False)
                pdf.savefig()
                plt.close(fig)
                
                fig =  plt.figure(figsize=(6,4), dpi= 300)
                counter: Dict[int, int] = dict(sorted(Counter(stats.n_actual2predicted).items()))
                #print(counter)
                plt.bar(counter.keys(), counter.values(), edgecolor="black")
                #print(np.arange(0, math.ceil(max(counter.values()))+1, math.ceil((math.ceil(max(counter.values()))+1)/5)))
                #print(np.arange(0, math.ceil(max(map(int, counter.keys())))+1, 1))
                plt.yticks(np.arange(0, roundup(max(counter.values()), 1000)+1, roundup(roundup(max(counter.values()), 1000)/5., 5)))
                plt.xticks([])  
                #plt.xticks(np.arange(0, math.ceil(max(map(int, counter.keys())))+1, 1))
                plt.title('Actual x-overs\nwith N overlapping predicted regions')
                plt.xlabel('N ACTUAL x-overs')
                plt.ylabel('N overlapping PREDICTED regions')
                sns.despine(top=True, right=True, left=False, bottom=False)
                plt.table(cellText=[list(counter.values())],
                      rowLabels=["xovers"],
                      colLabels=list(counter.keys()),
                      loc='bottom',colLoc='center', rowLoc='center')
                pdf.savefig()
                plt.close(fig)
                
                fig =  plt.figure(figsize=(6,4), dpi= 300)
                counter: Dict[int, int] = dict(sorted(Counter(stats.n_predicte2actual).items()))
                #print(counter)
                plt.bar(counter.keys(), counter.values(), edgecolor="black")
                plt.xticks([])  
                #print(np.arange(0, math.ceil(max(counter.values()))+1, math.ceil((math.ceil(max(counter.values()))+1)/5)))
                #print(np.arange(0, math.ceil(max(map(int, counter.keys())))+1, 1))
                plt.yticks(np.arange(0, roundup(max(counter.values()), 1000)+1, roundup(roundup(max(counter.values()), 1000)/5., 5)))
                plt.xticks(np.arange(0, math.ceil(max(map(int, counter.keys())))+1, 1))
                plt.title('Predicted regions\nwith N overlapping actual x-overs')
                plt.xlabel('N PREDICTED regions overlaps')
                plt.ylabel('N overlapping ACTUAL x-over')
                sns.despine(top=True, right=True, left=False, bottom=False)
                plt.table(cellText=[list(counter.values())],
                      rowLabels=["xovers"],
                      colLabels=list(counter.keys()),
                      loc='bottom',colLoc='center', rowLoc='center')
                pdf.savefig()
                plt.close(fig)
                #
                # #probabity range for markers
                # fig =  plt.figure(figsize=(6,4), dpi= 300)
                # plt.hist([b for b in np.concatenate([x for x in crossprobs_pat.data] ) if b > 0], bins='auto', edgecolor="black")
                # plt.title('Distribution of of non-zero crossover p̂\nin simulated paternal haplotype')
                # plt.xlabel('crossover p̂')
                # plt.ylabel('observed in simulations')
                # sns.despine(top=True, right=True, left=False, bottom=False)
                # pdf.savefig()
                # plt.close(fig)
                #
                # fig =  plt.figure(figsize=(6,4), dpi= 300)
                # plt.hist([b for b in np.concatenate([x for x in crossprobs_mat.data] ) if b > 0], bins='auto', edgecolor="black")
                # plt.title('Distribution of non-zero crossover probabilities in simulated maternal haplotype')
                # plt.xlabel('crossover p̂')
                # plt.ylabel('observed in simulations')
                # sns.despine(top=True, right=True, left=False, bottom=False)
                # pdf.savefig()  
                # plt.close(fig)
                #
                # #crossover events per simulation
                # fig =  plt.figure(figsize=(6,4), dpi= 300)
                # plt.hist([x for x in crossprobs_pat.sum(axis=0).tolist()], bins='auto', edgecolor="black")
                # plt.title('Distribution of n crossovers per sim\nin simulated paternal haplotype')
                # plt.xlabel('n crossovers on chromosome')
                # plt.ylabel('observed in simulations')
                # sns.despine(top=True, right=True, left=False, bottom=False)
                # pdf.savefig()  
                # plt.close(fig)
                #
                # fig =  plt.figure(figsize=(6,4), dpi= 300)
                # plt.hist([x for x in crossprobs_mat.sum(axis=0).tolist()], bins='auto', edgecolor="black")
                # plt.title('Distribution of n crossovers per sim\nin simulated maternal haplotype')
                # plt.xlabel('n crossovers on chromosome')
                # plt.ylabel('observed in simulations')
                # sns.despine(top=True, right=True, left=False, bottom=False)
                # pdf.savefig()  
                # plt.close(fig)
                
                # print("plotting subsample statistics")
                # arrayBase_pat = [x[0]/n_sims_pat for x in crossprobs_pat.sum(axis=1).tolist()]
                # arrayBase_mat = [x[0]/n_sims_mat for x in crossprobs_mat.sum(axis=1).tolist()]
                # idx = np.arange(0,crossprobs_pat.shape[1])
                # nsamples = 25
                # increment = math.ceil(crossprobs_pat.shape[1]/nsamples)
                # xaxis = [x for x in np.arange(increment, crossprobs_pat.shape[1]+1, increment)]
                # yaxis_pat = list()
                # yaxis_mat = list()
                # for n in xaxis:
                    # #print("subsample %s from %s " % (n, crossprobs_pat.shape[1]))
                    # data_pat = crossprobs_pat[:, np.random.choice(idx, n, replace=False)]
                    # data_mat = crossprobs_mat[:, np.random.choice(idx, n, replace=False)]
                    # data_pat = [x[0]/n for x in data_pat.sum(axis=1).tolist()]
                    # data_mat = [x[0]/n for x in data_mat.sum(axis=1).tolist()]
                    # print("calc stats %s %s " % (n, len(xaxis)))
                    # r, p = scipy.stats.pearsonr(arrayBase_pat,data_pat)
                    # yaxis_pat.append(r)
                    # r, p = scipy.stats.pearsonr(arrayBase_mat,data_mat)
                    # yaxis_mat.append(r)
                    #
                # fig=plt.figure(figsize=(12,4), dpi= 300, facecolor='w', edgecolor='k')
                # plt.plot(xaxis, yaxis_pat, color="blue", lw=1)
                # plt.plot(xaxis, yaxis_mat, color="purple", lw=1)
                # plt.xlabel('n siblings simulated')
                # plt.ylabel('r value (pearsons correlation)\n subsampled from %s simulations' % crossprobs_pat.shape[1])
                # plt.title('correlation of subsampled simulations sets\nto genome probability x-over density')
                # pdf.savefig()
                # plt.close(fig)
