#!/usr/local/bin/python3
# encoding: utf-8
'''
utils.extract_founders -- find founders in pedigree

utils.extract_founders is a simple utility for finding pedigree founders which allows user to add contraints on sequenced individuals

It defines classes_and_methods

@author:     mhindle

@copyright:  2021 University of Edinburgh. All rights reserved.

@license:    GPL

@contact:    matthew.hindle@ed.ac.uk
'''

import sys
import os
import gzip
import csv
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import numpy as np
from pedigree.pedigree_dag import PedigreeDAG

__all__ = []
__version__ = 0.1
__date__ = '2021-11-03'
__updated__ = '2021-11-03'

DEBUG =0
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
        parser.add_argument("-p", "--pedigree", dest="pedigree", required=True, help="pedigree file")
        parser.add_argument("-s", "--sequencedtsv", dest="sequencedtsv", required=False, help="A space separated file where the first column are ids that are sequenced data that will be used to constrain the pedigree: i.e. can be a phased haplotype (with duplicate ids) or genotype file")
        parser.add_argument("-o", "--outfile", dest="outfile", required=False, help="An outfile file for founders, one line per id")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        
        # Process arguments
        args = parser.parse_args()

        pedigree = PedigreeDAG.from_file(args.pedigree)
        
        if args.sequencedtsv is not None:
            sequenced_individuals = list()
            if args.sequencedtsv.endswith(".gz"):
                with gzip.open(args.sequencedtsv, 'rt') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ')
                    for row in reader:
                        try:
                            sequenced_individuals.append(int(row[0]))
                        except ValueError as e:
                            sys.stderr.write(str(e))
            else:     
                with open(args.sequencedtsv, 'rt') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ')
                    for row in reader:
                        try:
                            sequenced_individuals.append(int(row[0]))
                        except ValueError as e:
                            sys.stderr.write(str(e))
                        
            pedigree = pedigree.get_subset(sequenced_individuals, balance_parents=False)
        
            founder_no_sire = [x for x in pedigree.males.union(pedigree.females) if x not in pedigree.kid2sire.keys()]
            founder_no_dam = [x for x in pedigree.males.union(pedigree.females) if x not in pedigree.kid2dam.keys()]
            founders = np.array([x for x in set(founder_no_sire).intersection(set(founder_no_dam))])
        
        if args.outfile is not None:
            with open(args.outfile, 'wt') as csvfile:
                for founder in founders:
                    csvfile.write(str(founder)+"\n")
        else:
            for founder in founders:
                print(str(founder))
        
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
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'utils.extract_founders_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())