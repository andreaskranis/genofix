
"""Provides functions to import data from real SNP panels.

The moduls also provides functions to validate the imported data to ensure
that are suitable for quicgsim.

Todo:
    * For module TODOs

"""

from .genotype import Genotype
from . import np, tqdm


def read_pedigree():
    pass




######################################################
# OPTION 1: Import genome and genotypes in two steps #
######################################################

def read_snps(genome,inFile,id_col=None,chr_col=1,cm_col=None,sep=",",header=True):
    """Read a file .

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    def get_elem(array,pos,typos=str,def_val=None,verbose=False):
        """Wraps the try/except functionality to emaulate the <dict>.get() behaviour"""
        if pos is not None:
            try:
                return typos(array[pos])
                print(array)
                if verbose:
                    print(f"**WARNING: unable to convert value in position {pos} to type {typos}")
            except IndexError:
                print(array)
                if verbose:
                    print(f"**WARNING: unable to retrieve value in position {pos} from array {array}")
        return def_val
    
    
    with open(inFile) as fin:
        if header:
            next(fin)

        for row in fin:
            tmp = row.strip().split(sep)
            snpid,chrom,cm_pos = get_elem(tmp,id_col), get_elem(tmp,chr_col,verbose=True), get_elem(tmp,cm_col,typos=float)
            if chrom:
                genome.add_variant(chrom,snpid,cm_pos)
            
        for chrom in genome.chroms:
            genome.chroms[chrom].finalise_chrom_configuration()


def read_real_haplos(inFile,genome,first_haplo='maternal',mv=9,sep=None,header=False):
    gens = {}
    _SWITCH = {0:1,1:0}
    strand = 0

    with open(inFile) as fin:
        if header:
            next(fin)
        for row in tqdm.tqdm(fin):
            tmp = row.strip().split(sep)
            tag,g = tmp[0],np.array(tmp[1:],dtype=int)
            if tag not in gens:
                gens[tag] = Genotype(genome.chroms)
                if first_haplo == 'maternal':
                    strand = gens[tag].maternal_strand
                elif first_haplo == 'paternal':
                    strand = gens[tag].paternal_strand
            else:
                strand = _SWITCH[strand]

            st_pos = 0
            for c in gens[tag].iterate_chroms():
                end_pos = st_pos + genome.chroms[c].nvars
                gens[tag].add_haplo_toStrand(c,_SWITCH[strand],g[st_pos:end_pos],mv=9)
                st_pos += genome.chroms[c].nvars
    return gens

########################################################
# OPTION 2: Import PLINK for both genome and genotypes #
########################################################








