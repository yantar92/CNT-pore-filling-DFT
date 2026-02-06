# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:08:20 2026

@author: thefo
"""

import numpy as np

from ase.io import read
from ase.constraints import FixAtoms
from ase.ga.data import PrepareDB
# from ase.ga.startgenerator import StartGenerator
# from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from airsspy import SeedAtoms, Buildcell
from ase import Atoms
import os

from ase.visualize import view
import argparse

def radius_atom_list(struc):
    
    pos_struc = struc.get_positions()
    
    center = np.array([struc.cell[0,0]/2, struc.cell[0,0]/2, 0.0])
    
    pos_struc -= center
    
    return np.array(((pos_struc[:, 0]**2) + (pos_struc[:, 1]**2))**(1/2))

def make_db_GA(number_atoms, size_seeds):
    
    db_file = 'GA_' + str(number_atoms) + '_Na_CNT_15_0.db'

    # Reading CNT from CONTCAR
    cnt = read('CONTCAR')

    cnt.translate(-np.array([cnt.cell[0, 0]/2, cnt.cell[1, 1]/2, 0.0]))

    pos_cnt = cnt.get_positions()

    x_min = min(pos_cnt[:, 0]); x_max = max(pos_cnt[:, 0])
    y_min = min(pos_cnt[:, 1]); y_max = max(pos_cnt[:, 1])

    cnt.cell[0, 0] = max([(x_max - x_min), (y_max - y_min)]) + 15
    cnt.cell[1, 1] = max([(x_max - x_min), (y_max - y_min)]) + 15

    cnt.translate([cnt.cell[0, 0]/2, cnt.cell[1, 1]/2, 0.0])

    # cnt.set_constraint(FixAtoms(mask=len(cnt) * [True]))

    radius_cnt = max(radius_atom_list(cnt))

    atom_numbers = number_atoms * [11]

    cnt.extend(Atoms('Na'))
    seed = SeedAtoms(cnt)

    seed.gentags.supercell = '1 1 1'
    seed.gentags.slack = 0.1
    seed.gentags.minsep = [5.0, {'C-C': 1, 'Na-Na': 3.59346, 'C-Na': 2}]
    seed.gentags.fix = True
    # seed.gentags.cylinder = radius = (tube.cell[1, 1] - 2 * vacuum)/2

    for atom in seed:
        if atom.symbol == 'C':
            atom.fix = True
            atom.posamp = 0
        else:
            atom.zamp = -1
            atom.xamp = radius_cnt
            atom.yamp = radius_cnt
            atom.adatom = True
            atom.num = number_atoms
            atom.position = [seed.cell[0, 0]/2, seed.cell[1, 1]/ 2, 0]
    print('\n'.join(seed.get_cell_inp_lines()))
    bc = Buildcell(seed)

    starting_population = []
    for idx in range(size_seeds):
        atoms = bc.generate(timeout=100)
        mask_atoms = [f=='C' for f in atoms.get_chemical_symbols()]
        atoms.set_constraint(FixAtoms(mask=mask_atoms))
        starting_population.append(atoms)
    #    atoms.write(f'POSCAR_{idx}')

    if os.path.isfile(db_file):
        os.remove(db_file)

    # create the database to store information in
    d = PrepareDB(
        db_file_name=db_file, simulation_cell=cnt, stoichiometry=atom_numbers
    )

    for a in starting_population:
        d.add_unrelaxed_candidate(a)
        
parser = argparse.ArgumentParser(description="Script to add Na into optimized CNT. Must be CONTCAR file into the same folder")
parser.add_argument("--number_atoms", type=int, required=True, help="Number of Na")
parser.add_argument("--size_seeds", type=int, default=20, help="Number of seeds")

args = parser.parse_args()

make_db_GA(args.number_atoms, args.size_seeds)

# number_Na = 12
# size_seeds = 20  


