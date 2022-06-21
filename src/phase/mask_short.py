#!/usr/local/bin/python2.7
# encoding: utf-8
'''
plotting.dot_plot -- shortdesc

plotting.dot_plot is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2021 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import sys
import os
import phase, phase.evaluate, quickgsim
import numpy as np
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import pathlib

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

__all__ = []
__version__ = 0.1
__date__ = '2021-11-16'
__updated__ = '2021-11-16'

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

    # Setup argument parser
    parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-r", "--reference_file", dest="reference_file", help=" ")
    parser.add_argument("-i", "--input_file", dest="input_file", help=" ")
    parser.add_argument("-s", "--snp_map_file", dest="snp_file", help=" ")
    parser.add_argument("-o", "--output", dest="output", required=True, help="output directory")
    parser.add_argument('-V', '--version', action='version', version=program_version_message)
    
    # Process arguments
    args = parser.parse_args()
    
    reference_file = args.reference_file
    input_file = args.input_file
    snpmap = args.snp_file
    output = args.output
    
    genome = phase.evaluate.importGenome(snpmap)
    
    gens_reference = quickgsim.importers.read_real_haplos(reference_file, 
                               genome, first_haplo='maternal',
                               mv=9, sep=' ', header=False, random_assign_missing=False)
    
    gens_input = quickgsim.importers.read_real_haplos(input_file, 
                               genome, first_haplo='maternal',
                               mv=9, sep=' ', header=False, random_assign_missing=False)
    kids = len(gens_reference.keys())
    snps = np.sum([len(genotype[0]) for c,genotype in gens_reference[list(gens_reference.keys())[0]].data.items()])
    
    mismatch_mat = np.empty((kids,snps),dtype=int)
    mismatch_pat = np.empty((kids,snps),dtype=int)
    
    print(mismatch_mat.shape)
    for i, (id, genotype_ref) in enumerate(gens_reference.items()):
        genotype_query = gens_input.get(id)
        
        difference_mat = None
        difference_pat = None
        
        for chromosome, g in genotype_ref.data.items():
            paternal_ref = g[genotype_ref.paternal_strand]
            paternal_query = genotype_query[chromosome][genotype_query.paternal_strand]
            
            if len(paternal_ref) > 0:
                diff_pat = np.array(np.not_equal(paternal_ref, paternal_query), dtype=int)
                diff_pat[paternal_query == 9] = 9
                diff_pat[paternal_ref == 9] = -9
                #print(Counter(paternal_query))
                if difference_pat is None:
                    difference_pat = diff_pat
                else:
                    difference_pat = np.concatenate((difference_pat,diff_pat))
                
                maternal_ref = g[genotype_ref.maternal_strand]
                maternal_query = genotype_query[chromosome][genotype_query.maternal_strand]
                
                diff_mat = np.array(np.not_equal(maternal_ref, maternal_query),dtype=int)
                diff_mat[maternal_query == 9] = 9
                diff_mat[maternal_ref == 9] = -9
                
                if difference_mat is None:
                    difference_mat = diff_mat
                else:
                    difference_mat = np.concatenate((difference_mat,diff_mat))
        mismatch_mat[i] = difference_mat
        mismatch_pat[i] = difference_pat
        
    print(mismatch_mat)
    
    values = np.unique(mismatch_mat.ravel())
    print(values)
    if 9 in values:
        colours =  colors.ListedColormap(["black","yellow","red"], name='from_list', N=None)
    else:
        colours =  colors.ListedColormap(["black","yellow"], name='from_list', N=None)
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    
    plot = plt.figure()
    im = plt.imshow(mismatch_mat, interpolation='none', cmap=colours)
    patches = [ mpatches.Patch(color=colours.colors[i], label="Value {l}".format(l=values[i]) ) for i in range(len(values)) ]
    #patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]

    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    
    plt.xlabel("SNP (N)")
    plt.ylabel("Individual (N)")
    plt.savefig("%s/maternal_strand.png" % output)
    plt.close()
    
    #colours =  colors.ListedColormap(["red","violet","blue"], name='from_list', N=None)
    
    plot = plt.figure()
    im = plt.imshow(mismatch_pat, interpolation='none', cmap=colours)
    patches = [ mpatches.Patch(color=colours.colors[i], label="Value {l}".format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    plt.xlabel("SNP (N)")
    plt.ylabel("Individual (N)")
    plt.savefig("%s/paternal_strand.png" % output)
    plt.close()
    
if __name__ == "__main__":
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'plotting.dot_plot_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())