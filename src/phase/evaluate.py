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
import os
import pathlib
import pandas as pd
import quickgsim as gsim
from collections import defaultdict, Counter
import gzip
import mgzip, pickle, psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from quickgsim import genotype, importers

from crosssim import crossoverdetection
from pedigree.pedigree_dag import PedigreeDAG
from utils.pickle import dumpToPickle

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
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        
        # Process arguments
        args = parser.parse_args()

        reference_file = args.reference
        print("reference %s" % reference_file)
        subject_files = args.inputs
        print("subjects %s" % subject_files)
        snpmap = args.snpmap
        pedigree_file = args.pedigree_file
        xover = args.xover
        output_dir = args.output
        
        pedigree = PedigreeDAG.from_file(pedigree_file)
        
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        genomein = pd.read_csv(snpmap, sep=' ', names = ["chrom", "snpid", "cm", "pos"])
        
        real_xovers = {}
        with gzip.open(xover, "rt") as xoverref:
            for line in xoverref.readlines():
                values = line.strip().split("\t")
                xovers = [x.split(":") for x in values[1:] if x != ""]
                real_xovers[int(values[0])] = {int(chro):list(map(int,pos.split(","))) for chro,_direct,pos in xovers}
        
        chromosome2snp = defaultdict(set)
        chromosomes = set([row["chrom"] for index, row in genomein.iterrows()])
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
        
        #dumpToPickle("%s/genome.pickle.gz" % output_dir,genome)
        
        
        reference_pickle_file = "%s/%s.pickle.gz" % (output_dir,reference_file)
        if os.path.exists(reference_pickle_file) and os.path.isfile(reference_pickle_file):
            print("Loading pickle haplotype reference: %s" % reference_pickle_file)
            with mgzip.open(reference_pickle_file, "rb", thread=psutil.cpu_count(logical = False)) as reference_pickle:
                gens_reference = pickle.load(reference_pickle)
        else:
            print("Reading haplotype reference: %s" % reference_file)
            gens_reference = importers.read_real_haplos(reference_file, 
                               genome, first_haplo='maternal',
                               mv=9, sep=' ', header=False, random_assign_missing=False)
            dumpToPickle(reference_pickle_file,gens_reference)
        
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
        plt.savefig('%s/barplot_states_%s.png' % (output_dir,reference_file))
        plt.close()
        
        subject_file_obj = {}
        for subject_file in subject_files:
            subject_pickle_file = "%s/%s.pickle.gz" % (output_dir,subject_file)
            if os.path.exists(subject_pickle_file) and os.path.isfile(subject_pickle_file):
                print("Loading pickle haplotype subject: %s" % subject_pickle_file)
                with mgzip.open(subject_pickle_file, "rb", thread=psutil.cpu_count(logical = False)) as subject_pickle:
                    gens_subject = pickle.load(subject_pickle)
            else:
                print("Reading haplotype subject: %s" % subject_file)
                gens_subject = importers.read_real_haplos(subject_file, 
                               genome, first_haplo='maternal', 
                               mv=9, sep=' ', header=False, random_assign_missing=False)
                dumpToPickle(subject_pickle_file, gens_subject)
            subject_file_obj[subject_file] = gens_subject
        
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
            mat = plt.plot(list(chromosomes), [difference_by_chr_mat[chro]/called_by_chr_mat[chro] for chro in chromosomes], 
                                 'o', color='pink');
            pat = plt.plot(list(chromosomes), [difference_by_chr_pat[chro]/called_by_chr_pat[chro] for chro in chromosomes],
                                  'o', color='blue');
            plt.ylabel('pc error in called haplotype\n%s' % subject_file)
            plt.xlabel('Chromosome')
            plt.title('Haplotype errors from alphaimpute2')
            plt.savefig('%s/per_chromosome_errors_haplotype_calling_%s_vs_%s.png' % (output_dir,reference_file,subject_file))
            plt.close()
        
        print("stats for %s " % subject_file_obj.keys())
        for subject_file1, gens_subject1 in subject_file_obj.items():
            difference_by_chr_mat_1 = file2stats_maternal_difference[subject_file1]
            difference_by_chr_pat_1 = file2stats_paternal_difference[subject_file1]
            called_by_chr_mat_1 = file2stats_maternal_n[subject_file1]
            called_by_chr_pat_1 = file2stats_paternal_n[subject_file1]
            for subject_file2, gens_subject2 in [(subject_file2, gens_subject2) for subject_file2, gens_subject2 in subject_file_obj.items() if subject_file2 != subject_file1]:
                difference_by_chr_mat_2 = file2stats_maternal_difference[subject_file2]
                difference_by_chr_pat_2 = file2stats_paternal_difference[subject_file2]
                called_by_chr_mat_2 = file2stats_maternal_n[subject_file2]
                called_by_chr_pat_2 = file2stats_paternal_n[subject_file2]
            plt.figure()
            mat = plt.plot([difference_by_chr_mat_1[chro]/called_by_chr_mat_1[chro] for chro in chromosomes], 
                            [difference_by_chr_mat_2[chro]/called_by_chr_mat_2[chro] for chro in chromosomes], 
                                 'o', color='pink');
            pat = plt.plot([difference_by_chr_pat_1[chro]/called_by_chr_pat_1[chro] for chro in chromosomes], 
                            [difference_by_chr_pat_2[chro]/called_by_chr_pat_2[chro] for chro in chromosomes],
                                  'o', color='blue');
            plt.ylabel('pc error haplotype\n%s' % subject_file2)
            plt.xlabel('pc error haplotype\n%s' %  subject_file1)
            plt.title('Haplotype errors from alphaimpute2 \n %s vs %s' % (subject_file1, subject_file2))
            plt.savefig('%s/scatter_pcerrors_haplotype_calling_%s_vs_%s.png' % (output_dir,subject_file1,subject_file2))
            plt.close()
            
            plt.figure()
            mat = plt.plot(list(chromosomes),np.array([difference_by_chr_mat_1[chro] for chro in chromosomes])-np.array([difference_by_chr_mat_2[chro] for chro in chromosomes]), 
                                 'o', color='pink');
            pat = plt.plot(list(chromosomes),np.array([difference_by_chr_pat_1[chro] for chro in chromosomes])-np.array([difference_by_chr_pat_2[chro] for chro in chromosomes]),
                                  'o', color='blue');
            plt.ylabel('difference in errors haplotype\n%s-%s' % (subject_file1,subject_file2))
            plt.xlabel('Chromosome' )
            plt.title('Haplotype errors from alphaimpute2 \n %s vs %s' % (subject_file1, subject_file2))
            plt.axhline(y=0, color='k')
            plt.savefig('%s/scatter_errors_haplotype_calling_%s_vs_%s.png' % (output_dir,subject_file1,subject_file2))
            plt.close()
        
        print("stats for crossovers ")
        pc_predicted_real = []
        pc_real_predicted = []
        lengths_list = []
        for kid in pedigree.males.union(pedigree.females):
            sire, dam = pedigree.get_parents(kid)
            if sire is not None and dam is not None:
                if str(sire) in gens_reference.keys() and str(dam) in gens_reference.keys() and str(kid) in gens_reference.keys():
                    crossovers = crossoverdetection.predictCrossoverRegions(gens_reference[str(kid)], gens_reference[str(sire)], gens_reference[str(dam)])
                    #print("crossovers: %s" % crossovers)
                    for chro, predictedxover in crossovers.items():
                        detected = [range(p,p+(leng+1)) for p,leng in zip(predictedxover[1][0], predictedxover[1][1])]
                        true_ones = [np.any([real in d for real in real_xovers[kid][chro]]) for d in detected]
                        pc_predicted_real.append(
                            np.sum(true_ones)/ len(detected))
                        pc_real_predicted.append(
                            np.sum([np.any([real in d for d in detected]) for real in real_xovers[kid][chro]])/len(real_xovers[kid][chro]))
                        lengths_list.extend(predictedxover[1][1][true_ones])
                        #print("chr %s" % chro)
                        #paternalxover = [str(p)+"("+str(leng)+")" for p,leng in zip(predictedxover[1][0], predictedxover[1][1])]
                        #print("paternalxover: %s" % ",".join(map(str,paternalxover)))
                        #print("real_xovers: %s" % ",".join(map(str,real_xovers[kid][chro])))
        
        plt.figure()
        n, bins, patches = plt.hist(pc_predicted_real, bins=100, density=False, facecolor='g', alpha=0.75)
        plt.xlabel('pc predicted real')
        plt.ylabel('Count')
        plt.title('Histogram of percent xover predictions that are real')
        plt.grid(True)
        plt.xlim(0, 1)
        plt.savefig('%s/histogram_percent_xover_predictions_real_%s.png' % (output_dir,reference_file))
        plt.close()
        
        plt.figure()
        n, bins, patches = plt.hist(pc_real_predicted, bins=100, density=False, facecolor='g', alpha=0.75)
        plt.xlabel('pc real predicted')
        plt.ylabel('Count')
        plt.title('Histogram of percent real xover that are predicted')
        plt.grid(True)
        plt.xlim(0, 1)
        plt.savefig('%s/histogram_percent_xover_real_predicted_%s.png' % (output_dir,reference_file))
        plt.close()
        
        plt.figure()
        n, bins, patches = plt.hist(lengths_list, bins=100, density=False, facecolor='g', alpha=0.75)
        plt.xlabel('length of predicted region (i.e. ambiguity)')
        plt.ylabel('Count')
        plt.title('Histogram of lengths of true predicted regions')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig('%s/histogram_percent_xover_predictions_lengths_%s.png' % (output_dir,reference_file))
        plt.close()
        
        for subject_file, gens_subject in subject_file_obj.items():
            print("stats for crossovers %s" % subject_file)
            pc_predicted_real = []
            pc_real_predicted = []
            lengths_list = []
            for kid in pedigree.males.union(pedigree.females):
                sire, dam = pedigree.get_parents(kid)
                if sire is not None and dam is not None:
                    if str(sire) in gens_subject.keys() and str(dam) in gens_subject.keys() and str(kid) in gens_subject.keys():
                        crossovers = crossoverdetection.predictCrossoverRegions(gens_subject[str(kid)], gens_subject[str(sire)], gens_subject[str(dam)])
                        #print("crossovers: %s" % crossovers)
                        for chro, predictedxover in crossovers.items():
                            detected = [range(p,p+(leng+1)) for p,leng in zip(predictedxover[1][0], predictedxover[1][1])]
                            true_ones = [np.any([real in d for real in real_xovers[kid][chro]]) for d in detected]
                            pc_predicted_real.append(
                                np.sum(true_ones)/ len(detected))
                            pc_real_predicted.append(
                                np.sum([np.any([real in d for d in detected]) for real in real_xovers[kid][chro]])/len(real_xovers[kid][chro]))
                            lengths_list.extend(predictedxover[1][1][true_ones])
            
            plt.figure()
            n, bins, patches = plt.hist(pc_predicted_real, bins=100, density=False, facecolor='g', alpha=0.75)
            plt.xlabel('pc predicted real')
            plt.ylabel('Count')
            plt.title('Histogram of percent xover predictions that are real')
            plt.grid(True)
            plt.xlim(0, 1)
            plt.savefig('%s/histogram_percent_xover_predictions_real_%s.png' % (output_dir,subject_file))
            plt.close()
            
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