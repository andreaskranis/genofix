
"""Provides functions to import data from real SNP panels.

The moduls also provides functions to validate the imported data to ensure
that are suitable for quicgsim.

Todo:
    * For module TODOs

"""

from .genotype import Genotype
from . import np, tqdm
import gzip

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


def read_real_haplos(inFile, genome, first_haplo='maternal', snp_order=None,mv=9,sep=None,header=False,random_assign_missing=True):
    gens = {}
    _SWITCH = {0:1,1:0}
    strand = 0
    first_haplo_mat = True if first_haplo in ['maternal','m', 'mat', 'f'] else False
    if first_haplo_mat :
        print("Maternal First")
    else:
        print("Paternal First")
    if snp_order is not None:
        print("snp_order: %s" % snp_order[1:5])
    observed_order = None
    with gzip.open(inFile,"rt") if inFile.endswith("gz") else open(inFile,"rt") as fin:
        if header:
            observed_order = [x for x in next(fin).strip().replace("#","").split(sep)]
            observed_order = [x for x in observed_order if x != "" and x != "ID"]
            indexorder = np.array([observed_order.index(x) for x in snp_order if x in observed_order])
            print("observed_order: %s" % observed_order[1:5])
            print("indexorder: %s" % indexorder[1:5])
        for row in tqdm.tqdm(fin):
            tmp = row.strip().split(sep)
            tag,g = int(tmp[0]),np.array(tmp[1:],dtype=np.ushort)
            if snp_order is not None and observed_order is not None:
                g = g[indexorder]
            if tag not in gens:
                gens[tag] = Genotype(genome.chroms)
                if first_haplo_mat:
                    strand = gens[tag].maternal_strand
                else:
                    strand = gens[tag].paternal_strand
            else:
                strand = _SWITCH[strand]
            if tag == 4004165532145:
                print("%s %s %s " % (tag, strand, g[0:9]))
            st_pos = 0
            for c in gens[tag].iterate_chroms():
                end_pos = st_pos + genome.chroms[c].nvars
                gens[tag].add_haplo_toStrand(c,_SWITCH[strand],g[st_pos:end_pos],mv=mv, random_assign_missing=random_assign_missing)
                st_pos += genome.chroms[c].nvars
    return gens

def read_real_genos(inFile, genome,mv=9,sep=None,header=False):
    gens = {}
    rs = np.random.Generator(np.random.PCG64(1234)) 
    with gzip.open(inFile,"rt") if inFile.endswith("gz") else open(inFile,"rt") as fin:
        if header:
            next(fin)
        for row in tqdm.tqdm(fin):
            tmp = row.strip().split(sep)
            tag,g = int(tmp[0]),np.array(tmp[1:],dtype=np.ushort)
            gens[tag] = Genotype(genome.chroms)
            g = np.array([p if p != mv else rs.integers(low=0, high=3) for p in g],dtype=np.ushort)
            maternal = np.array([0 if x==0 else 1 if x==2 else rs.integers(low=0, high=2) for x in g],dtype=np.ushort)
            paternal = g-maternal
            if not np.equal(maternal+paternal, g).all():
                raise Exception("error in splitting haplotypes randomly")
            if np.greater(maternal, 1).any():
                raise Exception("maternal haplotype greater than 1????")
            if np.greater(paternal, 1).any():
                raise Exception("paternal haplotype greater than 1????")
            if np.greater(maternal+paternal, 2).any():
                raise Exception("genotype greater than 2????")
            
            st_pos = 0
            for c in gens[tag].iterate_chroms():
                end_pos = st_pos + genome.chroms[c].nvars
                gens[tag].add_haplo_toStrand(c,gens[tag].maternal_strand,maternal[st_pos:end_pos],mv=mv, random_assign_missing=False)
                st_pos += genome.chroms[c].nvars
            
            st_pos = 0
            for c in gens[tag].iterate_chroms():
                end_pos = st_pos + genome.chroms[c].nvars
                gens[tag].add_haplo_toStrand(c,gens[tag].paternal_strand,paternal[st_pos:end_pos],mv=mv, random_assign_missing=False)
                st_pos += genome.chroms[c].nvars
    return gens
########################################################
# OPTION 2: Import PLINK for both genome and genotypes #
########################################################








