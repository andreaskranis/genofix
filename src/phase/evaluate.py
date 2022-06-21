#!/usr/local/bin/python2.7
# encoding: utf-8
'''
phase.evaluate -- shortdesc

phase.evaluate is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2021 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

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

__all__ = []
__version__ = 0.1
__date__ = '2021-10-11'
__updated__ = '2021-10-11'

DEBUG = 0
TESTRUN = 0
PROFILE = 0

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

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
  Copyright 2021 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-m", "--snpmap", dest="snpmap", required=True, help="snp tsv file with columns \"chrom\", \"snpid\", \"cm\", \"pos\" where chrom is an integer")
        parser.add_argument("-r", "--reference", dest="reference", required=True, help="reference phase input file")
        parser.add_argument("-i", "--inputs", dest="inputs", required=True, help="phase input file/s", nargs='+')
        parser.add_argument("-o", "--output", dest="output", required=True, help="output directory")
        parser.add_argument("-p", "--pedigree", dest="pedigree_file", required=True, help="pedigree input file for triplets in crossover detection")
        parser.add_argument("-x", "--xover", dest="xover", required=True, help="xover input file with reference xover sites")
        parser.add_argument("-s", "--snps", dest="snps", required=False, type=int, default=sys.maxsize, help="")
        parser.add_argument("-e", "--exclusions", dest="exclusions", required=False, default=None, help="kid id exclusions")
        parser.add_argument("-f", "--flankmin", dest="min_flank_phase", nargs='+', required=False, type=int, default=None, help="")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        
        # Process arguments
        args = parser.parse_args()
        snps_n = args.snps
        reference_file = args.reference
        print("reference %s" % reference_file)
        subject_files = args.inputs
        print("subjects %s" % subject_files)
        snpmap = args.snpmap
        pedigree_file = args.pedigree_file
        xover = args.xover
        output_dir = args.output
        min_flank_phase=args.min_flank_phase
        pedigree = PedigreeDAG.from_file(pedigree_file)
        exclusions = args.exclusions
        
        print("min_flank_phase %s" % min_flank_phase)
        
        if exclusions is not None:
            with open(exclusions, "rt") as fh:
                exclusion_list = [int(x.strip()) for x in fh.readlines() if not x.startswith('#')]
            #print(exclusion_list)
        else:
            exclusion_list = []
        
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        real_mat_xovers = {}
        real_pat_xovers = {}
        with gzip.open(xover, "rt") as xoverref:
            for line in xoverref.readlines():
                values = line.strip().split("\t")
                id_kid = values[0]
                mat_p, pat_p = [_.start() for _ in re.finditer(id_kid, line)]
                mat_xovers = line.strip()[mat_p+len(id_kid)+1:pat_p]
                pat_xovers = line.strip()[pat_p+len(id_kid):]
                #print("mat %s %s~%s~ " % (len(id_kid), mat_p, mat_xovers))
                #print("pat %s %s~%s~ " % (len(id_kid), pat_p, pat_xovers))
                if len(mat_xovers) > 0:
                    xovers = [x.split(":") for x in mat_xovers.split("\t") if x != ""]
                    real_mat_xovers[int(values[0])] = {int(chro):[int(x) for x in pos.split(",") if int(x) <= snps_n] for chro,_direct,pos in xovers}
                if len(pat_xovers) > 0:
                    xovers = [x.split(":") for x in pat_xovers.split("\t") if x != ""]
                    real_pat_xovers[int(values[0])] = {int(chro):[int(x) for x in pos.split(",") if int(x) <= snps_n] for chro,_direct,pos in xovers}
        real_mat_xovers = {k:v for k,v in real_mat_xovers.items() if k not in exclusion_list}
        real_pat_xovers = {k:v for k,v in real_pat_xovers.items() if k not in exclusion_list}
        #dumpToPickle("%s/genome.pickle.gz" % output_dir,genome)
                
        genome = importGenome(snpmap)
        
        chromosomes = sorted(genome.chroms.keys())
        
        reference_pickle_file = "%s/%s.pickle.gz" % (output_dir,Path(reference_file).name)
        if os.path.exists(reference_pickle_file) and os.path.isfile(reference_pickle_file):
            print("Loading pickle haplotype reference: %s" % reference_pickle_file)
            with mgzip.open(reference_pickle_file, "rb", thread=psutil.cpu_count(logical = False)) as reference_pickle:
                gens_reference = pickle.load(reference_pickle)
        else:
            print("Reading haplotype reference: %s" % reference_file)
            gens_reference = importers.read_real_haplos(reference_file, 
                               genome, first_haplo='p',
                               mv=9, sep=' ', header=False, random_assign_missing=False)
            dumpToPickle(reference_pickle_file,gens_reference)
        
        #exclude exclusions...eg. founders
        
        print("mat %s ref %s" % (0,gens_reference[4004165532145][1][0][0:9]))
        print("pat %s ref %s" % (1,gens_reference[4004165532145][1][1][0:9]))
        print(gens_reference[4004165532145][1][0])
        print(gens_reference[4004165532145][1][1])
           
        #print("gens_reference: %s" % gens_reference.keys())
        
        ref_genotype_count = Counter()
        ref_genotype_count_chr = {int(chro):Counter() for chro in chromosomes}
        for kid, genotype in gens_reference.items():
            for chro, genotypes in genotype.data.items():
                maternal = Counter(genotypes[0])
                paternal = Counter(genotypes[1])
                ref_genotype_count = ref_genotype_count+maternal
                ref_genotype_count = ref_genotype_count+paternal
                ref_genotype_count_chr[chro] = ref_genotype_count_chr[chro]+maternal
                ref_genotype_count_chr[chro] = ref_genotype_count_chr[chro]+paternal
        
        print("by chromosome: %s" % (sorted(ref_genotype_count_chr.items())))
        print("all: %s" % (sorted(ref_genotype_count.items())))
        
        fig = plt.figure()
        states = [0, 1, 9]
        for chro in chromosomes:
            plt.bar(list(map(str,states)), [ref_genotype_count_chr[int(chro)][int(s)] for s in states])
        plt.savefig('%s/barplot_states_%s.png' % (output_dir,Path(reference_file).name))
        plt.close()
        
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
                               genome, first_haplo='p',
                               mv=9, sep=' ', header=False, random_assign_missing=False)
                dumpToPickle(subject_pickle_file, gens_subject)
            #only evaluate kids in reference
            gens_subject = {k:v for k,v in gens_subject.items() if k in gens_reference.keys()}
            subject_file_obj[Path(subject_file).name] = gens_subject
            print("mat %s gens_subject %s" % (0,gens_subject[4004165532145][1][0][0:9]))
            print("pat %s gens_subject %s" % (1,gens_subject[4004165532145][1][1][0:9]))
            
        
        file2stats_maternal_difference = {}
        file2stats_paternal_difference = {}
        
        file2stats_maternal_n = {}
        file2stats_paternal_n = {}
        
        print("stats for %s " % subject_file_obj.keys())
        for subject_file, gens_subject in subject_file_obj.items():
            ref_genotype_count = Counter()
            ref_genotype_count_chr = {int(chro):Counter() for chro in chromosomes}
            difference_by_chr_mat = {chro:0 for chro in chromosomes}
            difference_by_chr_pat = {chro:0 for chro in chromosomes}
            file2stats_maternal_difference[subject_file] = difference_by_chr_mat
            file2stats_paternal_difference[subject_file] = difference_by_chr_pat
            
            called_by_chr_mat = {chro:0 for chro in chromosomes}
            called_by_chr_pat = {chro:0 for chro in chromosomes}
            file2stats_maternal_n[subject_file] = called_by_chr_mat
            file2stats_paternal_n[subject_file] = called_by_chr_pat
            
            for kid, genotype in gens_subject.items():
                for chro, genotypes in genotype.data.items():
                    maternal_s = np.array(genotypes[0])
                    paternal_s = np.array(genotypes[1])
                    maternal_r = np.array(gens_reference[kid][chro][0])
                    paternal_r = np.array(gens_reference[kid][chro][1])
                    
                    na_neither_mat = np.logical_and(maternal_s != 9, maternal_r != 9)
                    difference_mat = np.not_equal(maternal_s[na_neither_mat],maternal_r[na_neither_mat])
                    #if np.sum(difference_mat) > 0:
                    #    print("mat  %s %s %s %s " % (subject_file, chro, np.sum(difference_mat), np.sum(na_neither_mat)))
                    difference_by_chr_mat[chro] = difference_by_chr_mat[chro]+np.sum(difference_mat)
                    called_by_chr_mat[chro] = called_by_chr_mat[chro]+np.sum(na_neither_mat)
                    
                    na_neither_pat = np.logical_and(paternal_s != 9, paternal_r != 9)
                    difference_pat = np.not_equal(paternal_s[na_neither_pat],paternal_r[na_neither_pat])
                    difference_by_chr_pat[chro] = difference_by_chr_pat[chro]+np.sum(difference_pat)
                    called_by_chr_pat[chro] = called_by_chr_pat[chro]+np.sum(na_neither_pat)
                    
                    maternal = Counter(maternal_s)
                    paternal = Counter(paternal_s)
                    ref_genotype_count = ref_genotype_count+maternal
                    ref_genotype_count = ref_genotype_count+paternal
                    ref_genotype_count_chr[chro] = ref_genotype_count_chr[chro]+maternal
                    ref_genotype_count_chr[chro] = ref_genotype_count_chr[chro]+paternal
                        
            print("diff by chromosome mat: %s" % difference_by_chr_mat)
            print("diff by chromosome pat: %s" % difference_by_chr_pat)
            print("by chromosome: %s" % (sorted(ref_genotype_count_chr.items())))
            #print("all: %s" % (sorted(ref_genotype_count.items())))
            
            fig = plt.figure()
            states = [0, 1, 9]
            for chro in chromosomes:
                plt.bar(list(map(str,states)), [ref_genotype_count_chr[int(chro)][int(s)] for s in states])
            plt.savefig('%s/barplot_states_%s.png' % (output_dir,subject_file))
            plt.close()
            
            fig = plt.figure()
            mat = plt.plot([chro for chro in chromosomes if called_by_chr_mat[chro] > 0], [difference_by_chr_mat[chro]/called_by_chr_mat[chro] for chro in chromosomes if called_by_chr_mat[chro] > 0], 
                                 'o', color='pink');
            pat = plt.plot([chro for chro in chromosomes if called_by_chr_pat[chro] > 0], [difference_by_chr_pat[chro]/called_by_chr_pat[chro] for chro in chromosomes if called_by_chr_pat[chro] > 0],
                                  'o', color='blue');
            plt.ylabel('pc error in called haplotype\n%s' % subject_file)
            plt.xlabel('Chromosome')
            plt.title('Haplotype errors from alphaimpute2')
            plt.savefig('%s/per_chromosome_errors_haplotype_calling_%s_vs_%s.png' % (output_dir,Path(reference_file).name,Path(subject_file).name))
            plt.close()
        
        #=======================================================================
        # print("stats for %s " % subject_file_obj.keys())
        # for subject_file1 in subject_file_obj.keys():
        #     difference_by_chr_mat_1 = file2stats_maternal_difference[subject_file1]
        #     difference_by_chr_pat_1 = file2stats_paternal_difference[subject_file1]
        #     called_by_chr_mat_1 = file2stats_maternal_n[subject_file1]
        #     called_by_chr_pat_1 = file2stats_paternal_n[subject_file1]
        #     for subject_file2 in [subject_file2 for subject_file2 in subject_file_obj.keys() if subject_file2 != subject_file1]:
        #         difference_by_chr_mat_2 = file2stats_maternal_difference[subject_file2]
        #         difference_by_chr_pat_2 = file2stats_paternal_difference[subject_file2]
        #         called_by_chr_mat_2 = file2stats_maternal_n[subject_file2]
        #         called_by_chr_pat_2 = file2stats_paternal_n[subject_file2]
        #         plt.figure()
        #         mat = plt.plot([difference_by_chr_mat_1[chro]/called_by_chr_mat_1[chro] for chro in chromosomes], 
        #                         [difference_by_chr_mat_2[chro]/called_by_chr_mat_2[chro] for chro in chromosomes], 
        #                              'o', color='pink');
        #         pat = plt.plot([difference_by_chr_pat_1[chro]/called_by_chr_pat_1[chro] for chro in chromosomes], 
        #                         [difference_by_chr_pat_2[chro]/called_by_chr_pat_2[chro] for chro in chromosomes],
        #                               'o', color='blue');
        #         plt.ylabel('pc error haplotype\n%s' % subject_file2)
        #         plt.xlabel('pc error haplotype\n%s' %  subject_file1)
        #         plt.title('Haplotype errors from alphaimpute2 \n %s vs %s' % (Path(subject_file1), Path(subject_file2)))
        #         plt.savefig('%s/scatter_pcerrors_haplotype_calling_%s_vs_%s.png' % (output_dir,Path(subject_file1), Path(subject_file2)))
        #         plt.close()
        #         
        #         plt.figure()
        #         mat = plt.plot(list(chromosomes),np.array([difference_by_chr_mat_1[chro] for chro in chromosomes])-np.array([difference_by_chr_mat_2[chro] for chro in chromosomes]), 
        #                              'o', color='pink');
        #         pat = plt.plot(list(chromosomes),np.array([difference_by_chr_pat_1[chro] for chro in chromosomes])-np.array([difference_by_chr_pat_2[chro] for chro in chromosomes]),
        #                               'o', color='blue');
        #         plt.ylabel('difference in errors haplotype\n%s-%s' % (subject_file1,subject_file2))
        #         plt.xlabel('Chromosome' )
        #         plt.title('Haplotype errors from alphaimpute2 \n %s vs %s' % (Path(subject_file1), Path(subject_file2)))
        #         plt.axhline(y=0, color='k')
        #         plt.savefig('%s/scatter_errors_haplotype_calling_%s_vs_%s.png' % (output_dir,Path(subject_file1), Path(subject_file2)))
        #         plt.close()
        # 
        #=======================================================================
        print("stats for crossovers ")
        pc_predicted_real = []
        pc_real_predicted = []
        real_x = []
        predicted_x = []
        lengths_list = []
        for kid in gens_reference.keys():
            sire, dam = pedigree.get_parents(kid)
            if sire is not None and dam is not None:
                if sire in gens_reference.keys() and dam in gens_reference.keys() and kid in gens_reference.keys():
                    mat_strand = 0
                    pat_strand = 1
                    crossovers = crossoverdetection.predictCrossoverRegions(gens_reference[kid], 
                                                                            gens_reference[sire], 
                                                                            gens_reference[dam],
                                                                            paternal_strand=pat_strand,
                                                                            maternal_strand=mat_strand)
                    #print("crossovers: %s" % crossovers)
                    for chro, predictedxover in crossovers.items():
                        if len(gens_reference[kid].data[chro][mat_strand]) > 0:
                            detected_mat = [range(p,p+(leng+2)) for p,leng in zip(predictedxover[mat_strand][0], predictedxover[mat_strand][1])]
                            detected_pat = [range(p,p+(leng+2)) for p,leng in zip(predictedxover[pat_strand][0], predictedxover[pat_strand][1])]
                            real_mat_xover_chr = real_mat_xovers[kid][chro] if chro in real_mat_xovers[kid] else np.array([])
                            real_pat_xover_chr = real_pat_xovers[kid][chro] if chro in real_pat_xovers[kid] else np.array([])
                            true_ones_mat = np.array([np.any([real in d for real in real_mat_xover_chr]) for d in detected_mat], dtype=bool)
                            true_ones_pat = np.array([np.any([real in d for real in real_pat_xover_chr]) for d in detected_pat], dtype=bool)
                            
                            #print("detected_mat %s" % detected_mat)
                            #print("detected_pat %s" % detected_pat)
                            #print("real_mat_xover_chr %s" % real_mat_xover_chr)
                            #print("real_pat_xover_chr %s" % real_pat_xover_chr)
                            
                            real_x.append(len(real_mat_xover_chr))
                            real_x.append(len(real_pat_xover_chr))
                            predicted_x.append(len(detected_mat))
                            predicted_x.append(len(detected_pat))
                                
                            if len(detected_mat) > 0:
                                pc_predicted_real.append(np.sum(true_ones_mat)/ len(detected_mat))
                            else:
                                pc_predicted_real.append(np.nan)
                                
                            if len(detected_pat) > 0:
                                pc_predicted_real.append(np.sum(true_ones_pat)/ len(detected_pat))
                            else:
                                pc_predicted_real.append(np.nan)
                            
                            def hasHit(values, truehits):
                                for hit in truehits:
                                    if hit in values:
                                        return(True)
                                return(False)
                            
                            if len(real_mat_xover_chr) > 0:
                                pc_real_predicted.append(
                                    np.sum([hasHit(real,real_mat_xover_chr) for real in detected_mat])/len(real_mat_xover_chr))
                            else:
                                pc_real_predicted.append(np.nan)

                            if len(real_pat_xover_chr) > 0:
                                pc_real_predicted.append(
                                    np.sum([hasHit(real,real_pat_xover_chr) for real in detected_pat])/len(real_pat_xover_chr))
                            else:
                                pc_real_predicted.append(np.nan)
                                
                            lengths_list.extend(np.array(predictedxover[mat_strand][1])[true_ones_mat])
                            lengths_list.extend(np.array(predictedxover[pat_strand][1])[true_ones_pat])
                            
                        #print("chr %s" % chro)
                        #paternalxover = [str(p)+"("+str(leng)+")" for p,leng in zip(predictedxover[1][0], predictedxover[1][1])]
                        #print("paternalxover: %s" % ",".join(map(str,paternalxover)))
                            #print("real_xovers: %s" % ",".join(map(str,real_xovers[kid][chro])))
        real_x = np.array(real_x, dtype=int)  
        predicted_x = np.array(predicted_x, dtype=int)
            
        sns.set_theme(style="ticks")
        print(Counter(real_x))
        print(Counter(predicted_x))
        sns.jointplot(x=real_x, y=predicted_x, 
                      kind="hex", color="#4CB391")
        plt.xlabel('predicted real (N)')
        plt.ylabel('real predicted (N)')
        plt.savefig('%s/REF_hex_histogram_%s.png' % (output_dir,Path(reference_file).name))
        plt.close()
        print(Counter(np.where(np.array(real_x) == 0, 0, np.log(real_x))))
        print(Counter(np.where(np.array(predicted_x) == 0, 0, np.log(predicted_x))))
        sns.set_theme(style="ticks")
        sns.jointplot(x=real_x, y=np.where(predicted_x == 0, 0, np.log(predicted_x)), 
                      kind="hex", color="#4CB391")
        plt.xlabel('real (N)')
        plt.ylabel('predicted (log N)')
        plt.savefig('%s/REF_hex_histogram_%s_log.png' % (output_dir,Path(reference_file).name))
        plt.close()
        
        sns.jointplot(x=real_x, y=predicted_x, 
                      kind="reg", color="#4CB391")
        plt.xlabel('predicted real (N)')
        plt.ylabel('real predicted (N)')
        plt.savefig('%s/REF_reg_histogram_%s.png' % (output_dir,Path(reference_file).name))
        plt.close()
                
        plt.figure()
        n, bins, patches = plt.hist(pc_predicted_real, bins=99, density=False, facecolor='g', alpha=0.75)
        plt.xlabel('pc predicted real')
        plt.ylabel('Count')
        plt.title('Histogram of percent xover predictions that are real')
        plt.grid(True)
        plt.xlim(0, 1)
        plt.yscale('log')
        plt.savefig('%s/REF_histogram_percent_xover_predictions_real_%s.png' % (output_dir,Path(reference_file).name))
        plt.close()
        
        plt.figure()
        n, bins, patches = plt.hist(pc_real_predicted, bins=100, density=False, facecolor='g', alpha=0.75)
        plt.xlabel('pc real predicted')
        plt.ylabel('Count')
        plt.title('Histogram of percent real xover that are predicted')
        plt.grid(True)
        plt.xlim(0, 1)
        plt.savefig('%s/REF_histogram_percent_xover_real_predicted_%s.png' % (output_dir,Path(reference_file).name))
        plt.close()
        
        plt.figure()
        n, bins, patches = plt.hist(lengths_list, bins=100, density=False, facecolor='g', alpha=0.75)
        plt.xlabel('length of predicted region (i.e. ambiguity)')
        plt.ylabel('Count')
        plt.title('Histogram of lengths of true predicted regions')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig('%s/REF_histogram_percent_xover_predictions_lengths_%s.png' % (output_dir,Path(reference_file).name))
        plt.close()
        
        per_file_real = OrderedDict()
        per_file_predicted = OrderedDict()
        per_file_false_positives = OrderedDict()
        per_file_false_negatives = OrderedDict()
        
        min_flank = 1
        
        for min_flank_errtolerance in min_flank_phase :
            for subject_file, gens_subject in subject_file_obj.items():
                print("stats for crossovers %s at flank %s " % (subject_file, min_flank))
                pc_predicted_real = []
                pc_real_predicted = []
                real_x = []
                predicted_x = []
                false_positives = []
                false_negatives = []
                lengths_list = []
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
                            #print("crossovers: %s" % crossovers)
                            for chro, predictedxover in crossovers.items():
                                if len(gens_subject[kid].data[chro][mat_strand]) > 0:
                                    detected_mat = [range(p-min_flank,p+(leng+1+min_flank)) for p,leng in zip(predictedxover[mat_strand][0], predictedxover[mat_strand][1])]
                                    detected_pat = [range(p-min_flank,p+(leng+1+min_flank)) for p,leng in zip(predictedxover[pat_strand][0], predictedxover[pat_strand][1])]
                                    real_mat_xover_chr = real_mat_xovers[kid][chro] if chro in real_mat_xovers[kid] else np.array([])
                                    real_pat_xover_chr = real_pat_xovers[kid][chro] if chro in real_pat_xovers[kid] else np.array([])
                                    true_ones_mat = np.array([np.any([real in d for real in real_mat_xover_chr]) for d in detected_mat], dtype=bool)
                                    true_ones_pat = np.array([np.any([real in d for real in real_pat_xover_chr]) for d in detected_pat], dtype=bool)
                                    
                                    real_x.append(len(real_mat_xover_chr))
                                    real_x.append(len(real_pat_xover_chr))
                                    predicted_x.append(len(detected_mat))
                                    predicted_x.append(len(detected_pat))
                                        
                                    false_positives.append(np.sum(np.logical_not(true_ones_mat)))
                                    false_positives.append(np.sum(np.logical_not(true_ones_pat)))
                                    
                                    false_negatives.append(len(real_mat_xover_chr)-np.sum(true_ones_mat))
                                    false_negatives.append(len(real_pat_xover_chr)-np.sum(true_ones_pat))
                                    
                                    # if (len(real_mat_xover_chr) > len(true_ones_mat)):
                                    #     for pred in detected_mat:
                                    #         if np.sum([x in pred for x in real_mat_xover_chr]) == 0 :
                                    #             print("predictedxover %s" % predictedxover)
                                    #             print("detected_mat %s " % (detected_mat))
                                    #             print("false pred %s %s " % (pred, real_mat_xover_chr))
                                    #             print("kid p: %s " % gens_subject[kid].data[chro][pat_strand][min(pred)-10:max(pred)+10])
                                    #             print("kid m: %s " % gens_subject[kid].data[chro][mat_strand][min(pred)-10:max(pred)+10])
                                    #             print("dam p: %s " % gens_subject[dam].data[chro][pat_strand][min(pred)-10:max(pred)+10])
                                    #             print("dam m: %s " % gens_subject[dam].data[chro][mat_strand][min(pred)-10:max(pred)+10])
                                    #             print("sir p: %s " % gens_subject[sire].data[chro][pat_strand][min(pred)-10:max(pred)+10])
                                    #             print("sir m: %s " % gens_subject[sire].data[chro][mat_strand][min(pred)-10:max(pred)+10])
                                     
                                    if len(detected_mat) > 0:
                                        pc_predicted_real.append(
                                            np.sum(true_ones_mat)/ len(detected_mat))
                                    else:
                                        pc_predicted_real.append(np.nan)
                                        
                                    if len(detected_pat) > 0:
                                        pc_predicted_real.append(
                                            np.sum(true_ones_pat)/ len(detected_pat))
                                    else:
                                        pc_predicted_real.append(np.nan)
                                    
                                    def hasHit(values, truehits):
                                        for hit in truehits:
                                            if hit in values:
                                                return(True)
                                        return(False)
                                    
                                    if len(real_mat_xover_chr) > 0:
                                        pc_real_predicted.append(
                                            np.sum([hasHit(real,real_mat_xover_chr) for real in detected_mat])/len(real_mat_xover_chr))
                                    else:
                                        pc_real_predicted.append(np.nan)
        
                                    if len(real_pat_xover_chr) > 0:
                                        pc_real_predicted.append(
                                            np.sum([hasHit(real,real_pat_xover_chr) for real in detected_pat])/len(real_pat_xover_chr))
                                    else:
                                        pc_real_predicted.append(np.nan)
                                        
                                    lengths_list.extend(np.array(predictedxover[mat_strand][1])[true_ones_mat])
                                    lengths_list.extend(np.array(predictedxover[pat_strand][1])[true_ones_pat])
            
                per_file_real[(subject_file,min_flank_errtolerance)] = real_x
                per_file_predicted[(subject_file,min_flank_errtolerance)] = predicted_x
                per_file_false_positives[(subject_file,min_flank_errtolerance)] = false_positives
                per_file_false_negatives[(subject_file,min_flank_errtolerance)] = false_negatives
            
                real_x = np.array(real_x, dtype=int)  
                predicted_x = np.array(predicted_x, dtype=int)
                
                sns.set_theme(style="ticks")
                sns.jointplot(x=real_x, y=predicted_x, 
                              kind="hex", color="#4CB391")
                plt.xlabel('real (N)')
                plt.ylabel('predicted (N)')
                plt.savefig('%s/hex_histogram_%s.png' % (output_dir,subject_file))
                plt.close()
                
                sns.set_theme(style="ticks")
                sns.jointplot(x=np.where(real_x != 0, np.log(real_x), 0), 
                              y=np.where(predicted_x != 0, np.log(predicted_x), 0), 
                          kind="hex", color="#4CB391")
                plt.xlabel('real (N)')
                plt.ylabel('predicted (N)')
                plt.savefig('%s/hex_histogram_%s_log.png' % (output_dir,subject_file))
                plt.close()
                
                print(Counter(np.array(predicted_x)))
                print(Counter(np.array(real_x)))
                
                sns.jointplot(x=real_x, y=predicted_x, kind="reg", color="#4CB391")#)
                plt.xlabel('real (N)')
                plt.ylabel('predicted (N)')
                plt.savefig('%s/reg_histogram_%s.png' % (output_dir,subject_file))
                plt.close()
                
                #===============================================================
                # plt.figure()
                # n, bins, patches = plt.hist(pc_predicted_real, bins=100, density=False, facecolor='g', alpha=0.75)
                # plt.xlabel('pc predicted real')
                # plt.ylabel('Count')
                # plt.title('Histogram of percent xover predictions that are real')
                # plt.grid(True)
                # plt.xlim(0, 1)
                # plt.yscale('log')
                # plt.savefig('%s/histogram_percent_xover_predictions_real_%s.png' % (output_dir,subject_file))
                # plt.close()
                #===============================================================
                
                plt.figure()
                n, bins, patches = plt.hist(pc_real_predicted, bins=100, density=False, facecolor='g', alpha=0.75)
                plt.xlabel('pc real predicted')
                plt.ylabel('Count')
                plt.title('Histogram of percent real xover that are predicted')
                plt.grid(True)
                plt.xlim(0, 1)
                plt.savefig('%s/histogram_percent_xover_real_predicted_%s.png' % (output_dir,subject_file))
                plt.close()
                
                plt.figure()
                n, bins, patches = plt.hist(lengths_list, bins=100, density=False, facecolor='g', alpha=0.75)
                plt.xlabel('length of predicted region (i.e. ambiguity)')
                plt.ylabel('Count')
                plt.title('Histogram of lengths of true predicted regions')
                plt.grid(True)
                plt.yscale('log')
                plt.savefig('%s/histogram_percent_xover_predictions_lengths_%s.png' % (output_dir,subject_file))
                plt.close()

        gens_reference = {k:v for k,v in gens_reference.items() if k not in exclusion_list}
        
        reference_xovers = [real_mat_xovers[k][1]+real_pat_xovers[k][1]  for k in real_mat_xovers.keys() if k in gens_reference.keys()]
        reference_xovers = [[c for c in x if c < 5000] for x in reference_xovers]
        n_reference_xovers = [len(x) for x in reference_xovers]
        
        largestValue = max(max(n_reference_xovers), max([max(x) for x in per_file_predicted.values()]))
        
        for err in [0.25,0.5,0.75,1,2,3,4] :
            #bins = np.linspace(-1, largestValue if largestValue < 15 else 15, largestValue if largestValue < 15 else 15)
            bins = np.linspace(-1, 15,17)
            print(bins)
            n,bins,patchs = plt.hist([x if x < 15 else 15 for x in n_reference_xovers], bins=bins, density=False, histtype="step", align="mid", color="black", label='reference')
            palette = sns.color_palette("Paired", n_colors= len(per_file_predicted))
            linstyle = {"reference": 'solid',"corrected":'dashed'}
            i=0
            for file, min_flank in per_file_predicted.keys():
                type = file.split("_")[1]
                errlevel = float(file.replace(".gz","").replace(".haplotypes","").replace("_data","").split("_")[5])
                if errlevel == err:
                    color = palette[i]
                    plt.hist([x if x < 15 else 15 for x in per_file_predicted[(file,min_flank)]], bins=bins, density=False, histtype="step", align="mid", color=color
                             , label=str(errlevel)+str(min_flank),linestyle=linstyle[type])
                    i+=1
            
            plt.legend(loc='upper right')
            #plt.xticks(rotation=45)
            plt.savefig('%s/crossoversperkid_histogram+_%s.svg' % (output_dir, err))
            plt.close()
        
        for file, min_flank in per_file_predicted.keys():
            type = file.split("_")[1]
            errlevel = float(file.replace(".gz","").replace(".haplotypes","").replace("_data","").split("_")[5])
            if min_flank == 0 and errlevel == 1 and type == "corrected":
                valuesB = [x if x < 15 else 15 for x in per_file_predicted[(file,min_flank)]]
                valuesA = [x if x < 15 else 15 for x in n_reference_xovers]
                m = np.zeros((16,16),dtype=int)
                for i,a in enumerate(valuesA):
                    b = valuesB[i]
                    m[b][a] = m[b][a]+1
        
        mpl.imshow(m, cmap='viridis')
        for (j,i),label in np.ndenumerate(m):
            #if label > 1000:
            label=str(round(label/1000,2))+"K"
            #plt.text(i,j,label,ha='center',va='center')
        mpl.colorbar()
        plt.savefig('%s/heat_overlap_histogram.svg' % (output_dir))
        plt.close()
        
        data = pd.DataFrame({
            "real_xovers":[np.sum(per_file_real[(file,min_flank)]) for file, min_flank in per_file_real.keys()],
            "predicted_xovers":[np.sum(per_file_predicted[(file,min_flank)]) for file, min_flank in per_file_real.keys()],
            "genotype_errors":[float(file.replace(".gz","").replace(".haplotypes","").replace("_data","").split("_")[5]) for file, min_flank in per_file_real.keys()],
            "false_positives":[np.sum(per_file_false_positives[(file,min_flank)]) for file, min_flank in per_file_real.keys()],
            "false_negatives":[np.sum(per_file_false_negatives[(file,min_flank)]) for file, min_flank in per_file_real.keys()],
            "xover_error_tolerance":[min_flank for file, min_flank in per_file_real.keys()],
            "type":[file.split("_")[1] for file, min_flank in per_file_real.keys()]
        })
        print(data)
    
        rcParams['figure.figsize'] = 10.4, 4.8
        sns.set(font_scale =1.5)
        sns.set_style("ticks")
        
        sns.lineplot(data=data, x="genotype_errors", y="false_positives", hue="xover_error_tolerance", 
                     style="type", markers=True, 
                     palette=sns.color_palette("rocket", n_colors=len(set([min_flank for file, min_flank in per_file_real.keys()]))))
        secax = plt.gca().secondary_yaxis('right', functions=(lambda x: (x/19505000)*5000, lambda x: (x/5000)*19505000))
        #plt.xticks(rotation=45)
        sns.despine()
        plt.savefig('%s/subjectFileErrors_false_positives.svg' % (output_dir))
        plt.close()
        
        rcParams['figure.figsize'] = 10.4, 4.8 
        sns.set(font_scale = 1.5)
        sns.set_style("ticks")
        sns.lineplot(data=data, x="genotype_errors", y="false_negatives", hue="xover_error_tolerance", 
                     style="type", markers=True, 
                     palette=sns.color_palette("rocket", n_colors=len(set([min_flank for file, min_flank in per_file_real.keys()]))))
        secax = plt.gca().secondary_yaxis('right', functions=(lambda x: (x/19505000)*5000, lambda x: (x/5000)*19505000))
        
        #plt.xticks(rotation=45)
        sns.despine()
        plt.savefig('%s/subjectFileErrors_false_negatives.svg' % (output_dir))
        plt.close()
            
        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
        sys.argv.append("-r")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'phase.evaluate_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())
    
                                    # if np.sum([hasHit(real,real_mat_xover_chr) for real in detected_mat]) == 0:
                                #     print("detected_mat %s" % detected_mat)
                                #     print("real_mat_xover_chr %s" % real_mat_xover_chr)
                                #     print("detected_pat %s" % detected_pat)
                                #     print("real_pat_xover_chr %s" % real_pat_xover_chr)#print("hasHit %s" % [hasHit(real,real_mat_xover_chr) for real in detected_mat])
                                #     print("ratio %s %s " % (len(real_mat_xover_chr), np.sum([hasHit(real,real_mat_xover_chr) for real in detected_mat])))
                                #     print(range(real_mat_xover_chr[0]-10,real_mat_xover_chr[0]+10))
                                #     region = range(real_mat_xover_chr[0]-10,real_mat_xover_chr[0]+10)
                                #     print(gens_reference[dam].data[chro][mat_strand][region])
                                #     print(gens_reference[dam].data[chro][pat_strand][region])
                                #     print(gens_reference[sire].data[chro][mat_strand][region])
                                #     print(gens_reference[sire].data[chro][pat_strand][region])
                                #     print(gens_reference[kid].data[chro][mat_strand][region])
                                #     print(gens_reference[kid].data[chro][pat_strand][region])
                                #     sys.exit()
                                