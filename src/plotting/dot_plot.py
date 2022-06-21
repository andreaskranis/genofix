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
from pathlib import Path

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import altair as alt


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
    parser.add_argument("-r", "--reference_file", dest="reference_file", required=True, help=" ")
    parser.add_argument("-i", "--input_files", dest="input_files", required=True, nargs='+',help=" ")
    parser.add_argument("-s", "--snp_map_file", dest="snp_file", required=True, help=" ")
    parser.add_argument("-o", "--output", dest="output", required=True, help="output directory")
    parser.add_argument("-e", "--exclusions", dest="exclusions", required=False, default=None, help="kid id exclusions")
    parser.add_argument('-V', '--version', action='version', version=program_version_message)
    
    # Process arguments
    args = parser.parse_args()
    
    reference_file = args.reference_file
    input_files = args.input_files
    snpmap = args.snp_file
    output = args.output
    exclusions = args.exclusions
    
    if exclusions is not None:
        with open(exclusions, "rt") as fh:
            exclusion_list = [int(x.strip()) for x in fh.readlines() if not x.startswith('#')]
        print(exclusion_list)
    else:
        exclusion_list = []
    
    with open(snpmap, "rt") as fh:
        snp_order = [x.strip().split(' ')[1] for x in fh.readlines() if not x.startswith('#')]
        snp_order = np.array([x for x in snp_order if x != "id"])
    
    print(snp_order[1:9])
    
    genome = phase.evaluate.importGenome(snpmap)
    
    gens_reference = quickgsim.importers.read_real_haplos(reference_file, 
                               genome, first_haplo='p',
                               mv=9, sep=' ', header=False, random_assign_missing=False)
    gens_reference = {k:v for k,v in gens_reference.items() if k not in exclusion_list}
    maternal_stats = {}
    paternal_stats = {}
    
    for input_file in input_files:
        Path(output).mkdir(parents=True, exist_ok=True)
        Path("%s/%s" % (output,Path(input_file).name)).mkdir(parents=True, exist_ok=True)
        
        gens_input = quickgsim.importers.read_real_haplos(input_file, 
                                   genome, first_haplo='m', snp_order=snp_order,
                                   mv=9, sep=' ', header=False, random_assign_missing=False)
        gens_input = {k:v for k,v in gens_input.items() if k in gens_reference.keys()}
        print(gens_input.keys())
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
                
                maternal_ref = g[genotype_ref.maternal_strand]
                maternal_query = genotype_query[chromosome][genotype_query.maternal_strand]
                    
                if len(paternal_ref) > 0:
                    diff_pat = np.array(np.not_equal(paternal_ref, paternal_query), dtype=int)
                    diff_pat[paternal_query == 9] = 9
                    diff_pat[paternal_ref == 9] = -9
                    #print(Counter(paternal_query))
                    if difference_pat is None:
                        difference_pat = diff_pat
                    else:
                        difference_pat = np.concatenate((difference_pat,diff_pat))
                    
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
        maternal_stats[Path(input_file).name] = (Counter(mismatch_mat.ravel()))
        if 9 in values:
            colours =  colors.ListedColormap(["black","red"], name='from_list', N=None)
        else:
            colours =  colors.ListedColormap(["black"], name='from_list', N=None)
        
        plot = plt.figure()
        im = plt.imshow(np.ma.masked_where(mismatch_mat == 0, mismatch_mat), interpolation='none', cmap=colours)
        print([(i,c) for i, c in enumerate(values) if i > 0])
        patches = [ mpatches.Patch(color=colours.colors[i-1], label="Value {l}".format(l=c) ) for i, c in enumerate(values) if i > 0]
        #patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
        plt.legend(handles=patches, loc=4, borderaxespad=0.)
        plt.xlabel("SNP (N)")
        plt.ylabel("Individual (N)")
        plt.savefig("%s/%s/imgplot_maternal_strand.png" % (output,Path(input_file).name))
        plt.close()
        
        values = np.unique(mismatch_pat.ravel())
        paternal_stats[Path(input_file).name] = (Counter(mismatch_pat.ravel()))
        if 9 in values:
            colours =  colors.ListedColormap(["black","red"], name='from_list', N=None)
        else:
            colours =  colors.ListedColormap(["black"], name='from_list', N=None)
        
        plot = plt.figure()
        im = plt.imshow(np.ma.masked_where(mismatch_pat == 0, mismatch_pat), interpolation='none', cmap=colours)
        patches = [ mpatches.Patch(color=colours.colors[i-1], label="Value {l}".format(l=c) ) for i, c in enumerate(values) if i > 0]
        plt.legend(handles=patches, loc=4, borderaxespad=0.)
        plt.xlabel("SNP (N)")
        plt.ylabel("Individual (N)")
        plt.savefig("%s/%s/imgplot_paternal_strand.png" % (output,Path(input_file).name))
        plt.close()
        
    for input_file in input_files:
        keys = Path(input_file).name
        
    df = pd.DataFrame({"Correct_phase": np.concatenate([[maternal_stats[Path(key).name][0],paternal_stats[Path(key).name][0]] for key in input_files]),
                    "Incorrect_phase" : np.concatenate([[maternal_stats[Path(key).name][1],paternal_stats[Path(key).name][1]] for key in input_files]),
                    "Unknown_phase": np.concatenate([[maternal_stats[Path(key).name][9],paternal_stats[Path(key).name][9]] for key in input_files]),
                    "Name":[a+" "+b for a,b in zip(np.repeat([Path(key).name.replace(".gz","").replace(".haplotypes","").split("_")[1] for key in input_files],2),["mat", "pat"]*len(input_files))],
                    "error":np.repeat([float(Path(key).name.replace("_data","").replace(".gz","").replace(".haplotypes","").split("_")[5]) for key in input_files],2),
                    "haplotype":["mat", "pat"]*len(input_files)
                    })
    
    #dflong = pd.wide_to_long(df, ["x"], i="Name", j="value", sep=":", suffix='\\w+')
    
    dflong = df.melt(["Name","error", "haplotype"], var_name="value",value_name="count")
    print(dflong)
    print(dflong.columns)
    print(dflong.index)
    
    chart = alt.Chart(dflong).mark_bar().encode(
    x=alt.Y('Name:N', title=None),
    y=alt.X('sum(count):Q', axis=alt.Axis(grid=False, title=None)),
    column=alt.Column('error:O', title=None),
    order=alt.Order('value:N', sort='ascending'),
    color=alt.Color('value:N', scale=alt.Scale(range=['#96ceb4', '#ffcc5c','#ff6f69']))
    )
       
    chart.configure_view(
        strokeOpacity=0    
    ).properties(
    width=800,
    height=300
    )
    
    chart.save("%s/stacked_bar_plot.html" % (output))
    
    chart = alt.Chart(dflong.loc[dflong.value != "Correct_phase",:]).mark_bar().encode(
    x=alt.Y('Name:N', title=None),
    y=alt.X('sum(count):Q', axis=alt.Axis(grid=False, title=None)),
    column=alt.Column('error:O', title=None),
    order=alt.Order('value:N', sort='ascending'),
    color=alt.Color('value:N', scale=alt.Scale(range=['#96ceb4', '#ffcc5c','#ff6f69']))
    )
       
    chart.configure_view(
        strokeOpacity=0    
    ).properties(
    width=800,
    height=300
    )
    
    chart.save("%s/stacked_bar_plot_errors.html" % (output))
    
    sns.set_style('ticks')
    sns.set_context("paper", font_scale = 2)
    g = sns.relplot(data=dflong.loc[dflong.value == "Incorrect_phase",:], x="error", y="count", hue="haplotype", kind="line", 
                    style="Name", legend="full", markers=True)
    plt.xticks([0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9])
    #plt.yticks([0,10, 100, 1000, 10000, 1000000],[0,10, 100, 1000, 10000, 1000000])
    #g.set(yscale="log")
    
    secax = plt.gca().secondary_yaxis('right', functions=(lambda x: (x/17480000)*100, lambda x: (x/100)*10017480000))
    secax.set_xlabel('phasing errors (%)')
    
    plt.savefig("%s/line_plot_errors.svg" % (output))
    
    print(maternal_stats)
    print(paternal_stats)
    
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