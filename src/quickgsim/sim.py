"""
"""
from . import Animal,np,tqdm


def create_founders(genders,gens,genome):
    """Can use either 'real' genotypes or artificial"""
    founders = {}
    a1 = set(genders.keys())
    a2 = set(gens.keys())
    print(f"Genotypes and gender info for {len(a2)} founders and {len(a1)} sexed individuals was received")
    
    common = a1.intersection(a2)
    print(f"** Note: {len(common)} instances of Animal() will be created")
    for a in common:
        founders[a] = Animal(a,genders[a],gens[a])
    return founders


def mate(sire,dam,kid_tag,genome,kid_sex=None):
    pg,pcr = genome.get_gamete(sire.genotype)
    mg,mcr = genome.get_gamete(dam.genotype)
    
    if not kid_sex:
        kid_sex = genome.rs.integers(1,3,dtype=int)
    gen = genome.get_zygote(pg,mg)
    return Animal(kid_tag,kid_sex,gen,pcr,mcr)  ##new kid


def drop_pedigree(pop,genome,ped,store_backend=None):
    for kid,sire,dam,sex in tqdm.tqdm(ped):
        progeny = mate(pop[sire],pop[dam],kid,genome,sex)
        pop[kid] = progeny
        if store_backend:
            store_backend.store_animal(kid,sire,dam,sex,progeny)
            
    if store_backend:
        store_backend.finalise()


