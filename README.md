# genofix
Using genomic data to fix pedigree and genotypes

## main entry points
###./genofix.py
Correct a genotype matrix 

###./simulate/simulate.py
Generate a gold standard genotype matrix (using random or by mating known founders and offspring to fill a pedigree), add known quantity of genotyping errors, and correct the genotype matrix (with statistics on precision and recall) 

###./utils/extract_founders.py
Create a list of founders from a pedigree, which can be contrained with a genotype file when a more complete pedigree is part sequenced

###./phase/evaluate.py
Compare two phase prediction outputs in AlphaImpute2 format

###./plotting/plot_rank_statistics.py
Create histograms and heat maps for rank statistics created by simulate.py
