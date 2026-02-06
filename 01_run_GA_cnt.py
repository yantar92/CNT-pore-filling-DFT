# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:12:08 2026

Use ase.ga to find ground state structures given
seed random structure DB.

@author: thefo
"""

from random import random
import os
from pathlib import Path
from contextlib import chdir
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet
from IMDgroup.gorun import gorun
from IMDgroup.gorun import slurm
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection
from ase.ga.offspring_creator import OperationSelector
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standardmutations import (
    MirrorMutation,
    RattleMutation,
)
from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from ase.io import write
# from ase.optimize import BFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Potcar
from pymatgen.io.vasp.inputs import Kpoints
import subprocess
import time
from ase.io import read
import sys
from ase.calculators.singlepoint import SinglePointCalculator
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
import argparse
import shutil


def relax_all_unrelaxed(da, vaspinput, directory):
    """Relax all unrelaxed structures in DA using VASPINPUT reference.
    Return 'converged' when everything has been relaxed.
    Return 'running' when the jobs were submitted and running.
    Return 'unconverged' when any of the structures failed to converge.
    """
    submitted_jobs = False
    for atoms in da.get_all_unrelaxed_candidates():
        path = atoms.info['data'].get('path', None)
        if path is None:
            idx = atoms.info['confid']
            generation = atoms.info['key_valid_pairs']['generation']
            atoms.info['data']['path'] = Path(f"{directory}/gen_{generation}_idx_{idx}").absolute()
        if not Path(path).is_dir():
            Path(path).mkdir()
            vaspinput.structure = AseAtomsAdaptor.get_structure(atoms)
            vaspinput.write_input(output_dir=path)
            print(f"Created new VASP input at {path}")
        vaspdir = IMDGVaspDir(path)
        if vaspdir.converged:
            tem = None
            try:
                tem = read(path / 'OUTCAR', index=-1)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                return False
            assert tem is not None
            energy = tem.get_potential_energy()
            forces = tem.get_forces()
            atoms.set_positions(a_tem.get_positions())
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            atoms.info['key_value_pairs']['raw_score'] = -atoms.get_potential_energy()
            da.add_relaxed_step(atoms)
            print(f"Added relaxed {path}")
        elif slurm.directory_queued_p(path):
            print(f"VASP still running in {path}")
            return 'running'
        elif 'CONTCAR' in vaspdir:
            print(f"VASP unconverged in {path}")
            return 'unconverged'
        else:
            with chdir(path):
                gorun.run()
                submitted_jobs = True
    if submitted_jobs:
        print("Submitted slurm jobs. Waiting for them to finish")
        return 'running'
    return 'converged'


def produce_new_generation(da, mutation_probability):
    """Add new generation to DA.
    """
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    population_size = len(list(da.c.select(generation=0)))
    # Load function to measure distance between atoms
    cnt = da.get_slab()
    all_atom_types = get_all_atom_types(cnt, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.8)

    comp = InteratomicDistanceComparator(
        n_top=n_to_optimize,
        pair_cor_cum_diff=0.015,
        pair_cor_max=0.7,
        dE=0.02,
        mic=False,
    )

    # Operations that will be performed for generate mutations
    pairing = CutAndSplicePairing(cnt, n_to_optimize, blmin)
    if n_to_optimize == 1:
        mutations = OperationSelector(
            [1.0],
            [
                RattleMutation(blmin, n_to_optimize),
            ],
        )
    else:
        mutations = OperationSelector(
            [1.0, 1.0],
            [
                MirrorMutation(blmin, n_to_optimize),
                RattleMutation(blmin, n_to_optimize),
            ],
        )

    population = Population(
        data_connection=da, population_size=population_size, comparator=comp
    )

    size = 0
    print(f'Creating new population of size {population_size}')
    while size < population_size:
        a1, a2 = population.get_two_candidates()
        a3, desc = pairing.get_new_individual([a1, a2])
        if a3 is None:
            continue
        da.add_unrelaxed_candidate(a3, description=desc)
        # Check if we want to do a mutation
        if random() < mutation_probability:
            a3_mut, desc = mutations.get_new_individual([a3])
            if a3_mut is not None:
                da.add_unrelaxed_step(a3_mut, desc)
                a3 = a3_mut
        size += 1


def run_ga(db_file, reference_vasp, mutation_probability=0.3):
    """Find ground states for seed structures from DB_FILE using generic algorithm.
    Use REFERENCE_VASP for INCAR setup
    1. Relax unrelaxed seed structures.
    2. Iteratively search GS using already known structures, relaxing one by one.
    """
    # Initialize the different components of the GA
    if not os.path.exists(db_file):
        print("File with seeds and candidates does not exist")
        sys.exit(1)

    shutil.copyfile(db_file, 'back_up.db')
    da = DataConnection(db_file)
    # Get atomic number of specie to optimize
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)

    def exit_saving_db():
        shutil.copyfile('back_up.db', db_file)
        sys.exit(1)

    vaspinput = IMDDerivedInputSet(
        directory=reference_vasp,
        user_incar_settings={"ISIF": 2, 'IBRION': 2},
        user_kpoints_settings={'grid_density': 6000}
        )

    while True:
        print(f"Generation {da.get_generation_number()}")
        status = relax_all_unrelaxed(da, vaspinput, f"{n_to_optimize}_Na")
        if status == 'unconverged':
            exit_saving_db()
        elif status == 'running':
            time.sleep(600)
        elif status == 'converged':
            produce_new_generation(da, mutation_probability)
        else:
            print('This should not happen')
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform single genetic mutation of structures, searching for GS.")
    parser.add_argument("structure_db", type=str, help="Path to structure DB")
    parser.add_argument("cnt_ref", type=str, help="Path to reference VASP calculation to get settings from")
    parser.add_argument("--mutation_probability", type=float, default=0.3, help="Mutation probability")
    args = parser.parse_args()
    run_ga(args.structure_db, args.cnt_ref, args.mutation_probability)
