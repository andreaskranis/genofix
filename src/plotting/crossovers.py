#!/usr/local/bin/python2.7
# encoding: utf-8
'''
plotting.crossovers -- shortdesc

plotting.crossovers is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2022 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import sys
import os

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import sys
import os, re
import pathlib
import pandas as pd
import quickgsim as gsim
from collections import defaultdict, Counter
import gzip
import mgzip, pickle, psutil
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from quickgsim import genotype, importers
import seaborn as sns
from matplotlib import rcParams

from crosssim import crossoverdetection
from pedigree.pedigree_dag import PedigreeDAG
from utils.pickle_util import dumpToPickle
from pathlib import Path

from collections import OrderedDict
import matplotlib.pyplot as mpl
import matplotlib.cm as cmap
from matplotlib.backends.backend_pdf import PdfPages

__all__ = []
__version__ = 0.1
__date__ = '2022-01-24'
__updated__ = '2022-01-24'

DEBUG = 0
TESTRUN = 0
PROFILE = 0


def importGenome(snpmap):
    genomein = pd.read_csv(snpmap, sep=' ', names = ["chrom", "snpid", "cm", "pos"])
        
    chromosome2snp = defaultdict(set)
    #chromosomes = set([row["chrom"] for index, row in genomein.iterrows()])
    chromosomesnps = {}
    for index, row in genomein.iterrows():
        if row["chrom"] not in chromosomesnps:
            chromosomesnps[row["chrom"]] = 1
        else:
            chromosomesnps[row["chrom"]]+=1
        chromosome2snp[row["chrom"]].add(row["snpid"])
    print("Chromosomes and n nsnps: %s" % chromosomesnps)
            
    genome = gsim.genome.Genome()
    
    for index, row in genomein.iterrows():
        if row["cm"] > 0:
            genome.add_variant(int(row["chrom"]), snpid=row["snpid"], cm_pos=float(row["cm"])+1)
    
    for chrom in genome.chroms.values():
        chrom.finalise_chrom_configuration()
    
    for chrom in genome.chroms.keys():
        genome.chroms[chrom].cm_pos = genome.chroms[chrom].cm_pos+abs(min(genome.chroms[chrom].cm_pos))
        genome.chroms[chrom].cm_pos = genome.chroms[chrom].cm_pos/sum(genome.chroms[chrom].cm_pos) 
        
    return(genome)

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by user_name on %s.
  Copyright 2022 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    # Setup argument parser
    parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
    parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--snpmap", dest="snpmap", required=True, help="snp tsv file with columns \"chrom\", \"snpid\", \"cm\", \"pos\" where chrom is an integer")
    parser.add_argument("-i", "--inputs", dest="inputs", required=True, help="phase input file/s", nargs='+')
    parser.add_argument("-o", "--output", dest="output", required=True, help="output directory")
    parser.add_argument("-p", "--pedigree", dest="pedigree_file", required=True, help="pedigree input file for triplets in crossover detection")
    parser.add_argument("-s", "--snps", dest="snps", required=False, type=int, default=sys.maxsize, help="")
    parser.add_argument("-e", "--exclusions", dest="exclusions", required=False, default=None, help="kid id exclusions")
    parser.add_argument("-f", "--flankmin", dest="min_flank_phase", nargs='+', required=False, type=int, default=[0], help="")
    parser.add_argument('-V', '--version', action='version', version=program_version_message)
    
    # Process arguments
    args = parser.parse_args()
    snps_n = args.snps
    subject_files = args.inputs
    print("subjects %s" % subject_files)
    snpmap = args.snpmap
    pedigree_file = args.pedigree_file
    output_dir = args.output
    min_flank_phase = args.min_flank_phase
    pedigree = PedigreeDAG.from_file(pedigree_file)
    exclusions = args.exclusions
    
    print("min_flank_phase %s" % min_flank_phase)
    
    if exclusions is not None:
        with open(exclusions, "rt") as fh:
            exclusion_list = [int(x.strip()) for x in fh.readlines() if not x.startswith('#')]
        #print(exclusion_list)
    else:
        exclusion_list = []
    
    print("exclusion_list %s" % exclusion_list)
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    genome = importGenome(snpmap)
    
    subject_file_obj = {}
    for subject_file in subject_files:
        subject_pickle_file = "%s/%s.pickle.gz" % (output_dir,Path(subject_file).name)
        if os.path.exists(subject_pickle_file) and os.path.isfile(subject_pickle_file):
            print("Loading pickle haplotype subject: %s" % subject_pickle_file)
            with mgzip.open(subject_pickle_file, "rb", thread=psutil.cpu_count(logical = False)) as subject_pickle:
                gens_subject = pickle.load(subject_pickle)
        else:
            print("Reading haplotype subject: %s" % subject_file)
            gens_subject = importers.read_real_haplos(subject_file, 
                           genome, first_haplo='m',
                           mv=9, sep=' ', header=False, random_assign_missing=False)
            dumpToPickle(subject_pickle_file, gens_subject)
        #only evaluate kids in reference
        gens_subject = {k:v for k,v in gens_subject.items()}
        subject_file_obj[Path(subject_file).name] = gens_subject
        print("mat %s gens_subject %s" % (0,gens_subject[4004165532145][1][0][0:9]))
        print("pat %s gens_subject %s" % (1,gens_subject[4004165532145][1][1][0:9]))
        
    
    for min_flank_errtolerance in min_flank_phase :
        print(min_flank_errtolerance)
        for subject_file, gens_subject in subject_file_obj.items():
            print("stats for crossovers %s at flank %s " % (subject_file, min_flank_errtolerance))
            pc_predicted_real = []
            pc_real_predicted = []
            real_x = []
            predicted_x = []
            false_positives = []
            false_negatives = []
            lengths_list = []
            
            def xtoString(chro, data):
                return(str(chro)+":"+",".join(map(str,data)))
            
            mat_strand = 0
            pat_strand = 1
                                 
            perChromosomeN_pat = {}
            perChromosomeN_mat = {}
                                  
            perChromosomeP_pat = {}
            perChromosomeP_mat = {}
            
            with gzip.open("%s/crossovers.txt.gz" % output_dir, "wt") as fout:
                for kid in gens_subject.keys():
                    sire, dam = pedigree.get_parents(kid)
                    if sire is not None and dam is not None:
                        if sire in gens_subject.keys() and dam in gens_subject.keys() and kid in gens_subject.keys():
                            mat_strand = 0
                            pat_strand = 1
                            crossovers = crossoverdetection.predictCrossoverRegions(gens_subject[kid]
                                                                                    , gens_subject[sire]
                                                                                    , gens_subject[dam]
                                                                                    , paternal_strand=pat_strand,
                                                                                    maternal_strand=mat_strand, 
                                                                                    min_flank_support=min_flank_errtolerance)
                            fout.write(str(kid))
                            for chro, predictedxover in crossovers.items():
                                if chro not in perChromosomeN_mat:
                                    perChromosomeN_mat[chro] = []
                                if chro not in perChromosomeP_mat:
                                    perChromosomeP_mat[chro] = {}
                                if kid not in perChromosomeP_mat[chro]:
                                    perChromosomeP_mat[chro][kid] = np.zeros(genome.chroms[chro].nvars, dtype=float)
                                    
                                perChromosomeN_mat[chro].append(len(predictedxover[mat_strand][0]))
                                detected_mat = [str(p)+"-"+str(p+(leng+1)) for p,leng in zip(predictedxover[mat_strand][0], predictedxover[mat_strand][1])]
                                fout.write("\t"+xtoString(chro,detected_mat))
                                for p,leng in zip(predictedxover[mat_strand][0], predictedxover[mat_strand][1]):
                                    perChromosomeP_mat[chro][kid][p:p+leng+1] = 1/(leng+1)
                                
                            fout.write("\n")
                            fout.write(str(kid))
                            for chro, predictedxover in crossovers.items():
                                if chro not in perChromosomeN_pat:
                                    perChromosomeN_pat[chro] = []
                                if chro not in perChromosomeP_pat:
                                    perChromosomeP_pat[chro] = {}
                                if kid not in perChromosomeP_pat[chro]:
                                    perChromosomeP_pat[chro][kid] = np.zeros(genome.chroms[chro].nvars, dtype=float)
                                 
                                perChromosomeN_pat[chro].append(len(predictedxover[pat_strand][0]))
                                detected_pat = [str(p)+"-"+str(p+(leng+1)) for p,leng in zip(predictedxover[pat_strand][0], predictedxover[pat_strand][1])]
                                fout.write("\t"+xtoString(chro,detected_pat))
                                for p,leng in zip(predictedxover[pat_strand][0], predictedxover[pat_strand][1]):
                                    perChromosomeP_pat[chro][kid][p:p+leng+1] = 1/(leng+1)
                            fout.write("\n")
            
            data = pd.DataFrame({
                "chromosome": [chro for chro,ns in perChromosomeN_pat.items() for n in ns]+[chro for chro,ns in perChromosomeN_mat.items() for n in ns],
                "strand": ["pat" for chro,ns in perChromosomeN_pat.items() for n in ns]+["mat" for chro,ns in perChromosomeN_mat.items() for n in ns],
                "N": [n for chro,ns in perChromosomeN_pat.items() for n in ns]+[n for chro,ns in perChromosomeN_mat.items() for n in ns]
            })
            rcParams['figure.figsize'] = 15.4, 4.8
            sns.set(font_scale =1.5)
            sns.set_style("ticks")
            ax = sns.stripplot(x="chromosome", y="N", hue="strand",data=data, palette="Set2", dodge=True)
            plt.gca().set_yscale('log')
            sns.despine()
            plt.savefig('%s/crossovers_per_chromosome.svg' % (output_dir))
            plt.close()

            def moving_average(x, w):
                return np.convolve(x, np.ones(w), 'valid') / w

            with PdfPages('%s/probabilities_allchromosomes_%s.pdf' % (output_dir,min_flank_errtolerance)) as pdf:
                for chro, kidshash_pat in perChromosomeP_pat.items():
                    print("plotting chromosome %s " % chro)
                    kidshash_mat = perChromosomeP_mat[chro]
                    kidshash_mat_v = kidshash_mat.values()
                    print(len(kidshash_mat_v))
                    kidshash_pat_v = kidshash_pat.values()
                    
                    genomeStack = np.stack(list(kidshash_pat_v)+list(kidshash_mat_v), axis=0)
                    
                    kidshash_mat_v2 = [x for x in kidshash_mat_v if sum(x) > 0 and sum(x) <= 5]
                    kidshash_pat_v2 = [x for x in kidshash_pat_v if sum(x) > 0 and sum(x) <= 5]
                    print(len(kidshash_mat_v2))
                    if len(kidshash_mat_v2) > 0 or len(kidshash_pat_v2) > 0 :
                        y = genomeStack.sum(0, dtype='float')/genomeStack.sum(dtype='float')
                        y2 = np.interp(y, (0, max(y)), (0, max(y)/sum(y)))
                        
                        genomeStack2 = np.stack(list(kidshash_pat_v2)+list(kidshash_mat_v2), axis=0)
                        y2 = genomeStack2.sum(0, dtype='float')/genomeStack2.sum(dtype='float')
                        y2 = np.interp(y, (0, max(y2)), (0, max(y2)/sum(y2)))
                        
                        x = range(len(y))
                        plt.plot(x, y, color="grey", linewidth=0.5)
                        
                        x = range(len(y2))
                        plt.plot(x, y2, color="blue", linewidth=0.5)
                        #plt.hlines(np.quantile(y2, 0.8), 0, len(y2), color="blue", linestyles="dashed", linewidth=0.5)
                        averaged = moving_average(y2, 50)
                        averaged = np.interp(averaged, (min(averaged), max(averaged)), (0, max(averaged)/sum(averaged)))
                        newx = np.array([x for x in range(len(averaged))], dtype='float')
                        newx= np.interp(newx, (0, max(newx)), (0, max(x)))
                        plt.plot(newx, averaged, color="red", linewidth=0.5)
                        plt.hlines(np.quantile(averaged, 0.8), 0, max(newx), color="red", linestyles="dashed", linewidth=0.25)
                        #plt.yscale("log")
                        plt.title('Chromosome %s xover probability filtered' % (chro))
                        plt.ylabel("xover P")
                        plt.xlabel("SNP position");
                        pdf.savefig()
                        plt.close()
    return 0

if __name__ == "__main__":
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'plotting.crossovers_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())