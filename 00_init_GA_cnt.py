# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:08:20 2026

Generate initial set of randomized inputs for CNT filled with Na.
Use AIRSS.

@author: thefo
"""

import os
import argparse
import warnings
import subprocess
import numpy as np
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
from ase.constraints import FixAtoms
from ase.ga.data import PrepareDB
from airsspy import SeedAtoms, Buildcell
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor


def radius_atom_list(struc):
    """Return a list of distances of atoms from central z axis.
    """
    pos_struc = struc.get_positions()
    center = np.array([struc.cell[0, 0]/2, struc.cell[0, 0]/2, 0.0])
    pos_struc -= center
    return np.array(((pos_struc[:, 0]**2) + (pos_struc[:, 1]**2))**(1/2))


def make_db_GA(cnt_dir, number_Na, size_seeds, vacuum=15):
    db_file = 'GA_' + str(number_Na) + '_Na_CNT.db'
    vaspdir = IMDGVaspDir(cnt_dir)
    assert vaspdir.converged
    cnt = AseAtomsAdaptor.get_atoms(vaspdir.structure)

    # Set vacuum
    cnt.translate(-np.array([cnt.cell[0, 0]/2, cnt.cell[1, 1]/2, 0.0]))
    pos_cnt = cnt.get_positions()
    x_min, x_max = min(pos_cnt[:, 0]), max(pos_cnt[:, 0])
    y_min, y_max = min(pos_cnt[:, 1]), max(pos_cnt[:, 1])
    cnt.cell[0, 0] = max([(x_max - x_min), (y_max - y_min)]) + vacuum
    cnt.cell[1, 1] = max([(x_max - x_min), (y_max - y_min)]) + vacuum
    cnt.translate([cnt.cell[0, 0]/2, cnt.cell[1, 1]/2, 0.0])

    print(f"Read relaxed CNT: {cnt}")

    # Compute radius to limit AIRSS distortions
    radius_cnt = max(radius_atom_list(cnt))

    # Add Na at 0,0,0 to be randomized by AIRSS
    cnt_origin = cnt.copy()
    cnt.extend(Atoms('Na'))
    seed = SeedAtoms(cnt)
    seed.gentags.supercell = '1 1 1'
    # Allow slightly smaller atom-atom distances
    seed.gentags.slack = 0.1
    # Minimum distances between atoms
    seed.gentags.minsep = [5.0, {'C-C': 1.4, 'Na-Na': 3.59346, 'C-Na': 2}]
    seed.gentags.fix = True
    # seed.gentags.cylinder = radius = (tube.cell[1, 1] - 2 * vacuum)/2
    for atom in seed:
        if atom.symbol == 'C':
            # All carbons remain in place
            atom.fix = True
            atom.posamp = 0
        else:
            # All Na can randomly move up to radius_cnt in x/y direction
            # z - any
            # atom.fix = False
            atom.zamp = -1
            atom.xamp = radius_cnt
            atom.yamp = radius_cnt
            # only matters for supercell, but keep for safety - from examples
            atom.adatom = True
            # Add number_Na Na atoms during randomization
            atom.num = number_Na
            # Na to be placed in the middle of the CNT and randomized from there
            atom.position = [seed.cell[0, 0]/2, seed.cell[1, 1]/ 2, 0]
    print('\n'.join(seed.get_cell_inp_lines()))
    bc = Buildcell(seed)

    starting_population = []
    for _ in range(size_seeds):
        atoms = bc.generate(timeout=100)
        # FIXME: Why??
        atoms.set_pbc(True)
        print(atoms)
        # 2026-03-23: Allowing Carbon relaxation.
        # mask_atoms = [f == 'C' for f in atoms.get_chemical_symbols()]
        # # We will not allow carbons to move during relaxation later.
        # atoms.set_constraint(FixAtoms(mask=mask_atoms))
        starting_population.append(atoms)

    if os.path.isfile(db_file):
        warnings.warn(f"Overwriting db file: {db_file}")
        os.remove(db_file)

    # create the database to store information in
    atom_numbers = number_Na * [11]  # 11 is atomic number of Na
    d = PrepareDB(
        db_file_name=db_file, simulation_cell=cnt_origin, stoichiometry=atom_numbers,
        population_size=len(starting_population),
    )

    for a in starting_population:
        d.add_unrelaxed_candidate(a)
    print(f"Generated structures saved to {db_file}")


def main(args):
    make_db_GA(args.cnt, args.number_Na, args.size_seeds, args.vacuum)
    # number_Na = 12
    # size_seeds = 20


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to add Na into optimized CNT")
    parser.add_argument("--cnt", type=str, required=True, help="Path to directory with relaxed CNT")
    parser.add_argument("--number_Na", type=int, required=True, help="Number of Na")
    parser.add_argument("--size_seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--vacuum", type=float, default=15.0, help="Vacuum to surround CNT with")

    args = parser.parse_args()
    try:
        subprocess.check_output(['buildcell', '-v'])
    except subprocess.CalledProcessError:
        print("Cannot run buildcell. Check installation.")
        exit(1)
    main(args)
