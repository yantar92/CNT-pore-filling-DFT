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
import numpy as np
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

GORUN_ARGS = argparse.Namespace(mark=True)

def relax_all_unrelaxed(da, vaspinput, directory):
    """Relax all unrelaxed structures in DA using VASPINPUT reference.
    Return 'converged' when everything has been relaxed.
    Return 'running' when the jobs were submitted and running.
    Return 'unconverged' when any of the structures failed to converge.
    """
    submitted_jobs = False
    has_unconverged = False
    for atoms in da.get_all_unrelaxed_candidates():
        atoms = atoms.copy()
        path = atoms.info['data'].get('path', None)
        if path is None:
            idx = atoms.info['confid']
            generation = atoms.info['key_value_pairs']['generation']
            path = Path(f"{directory}/gen_{generation}_idx_{idx}").absolute()
            atoms.info['data']['path'] = str(path)
        if not Path(path).is_dir():
            Path(path).mkdir(parents=True)
        relax_dir = Path(path) / "relax"
        scf_dir = Path(path) / "relax.SCF"

        if not relax_dir.is_dir():
            relax_dir.mkdir(parents=True)
            vaspinput.structure = AseAtomsAdaptor.get_structure(atoms)
            vaspinput.write_input(output_dir=relax_dir)
            print(f"Created new relax VASP input at {relax_dir}")
            with chdir(relax_dir):
                gorun.run(GORUN_ARGS)
                submitted_jobs = True
                continue

        if scf_dir.is_dir ():
            path = scf_dir
        else:
            path = relax_dir
        vaspdir = IMDGVaspDir(path)
        if vaspdir.converged and path == scf_dir:
            tem = None
            try:
                tem = read(path / 'OUTCAR', index=-1)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                return False
            assert tem is not None
            energy = tem.get_potential_energy()

            tem_forces = tem.get_forces()
            new_positions = np.zeros_like(atoms.positions)
            new_forces = np.zeros_like(tem_forces)
            for symbol in set(tem.get_chemical_symbols()):
                src_idx = [i for i, sym in enumerate(tem.get_chemical_symbols()) if sym == symbol]
                tgt_idx = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == symbol]
                for tgt, src in zip(tgt_idx, src_idx):
                    new_positions[tgt] = tem.positions[src]
                    new_forces[tgt] = tem_forces[src]

            atoms.set_positions(new_positions)
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=new_forces)
            atoms.info['key_value_pairs']['raw_score'] = -atoms.get_potential_energy()
            da.add_relaxed_step(atoms)
            print(f"Added relaxed {path}")
        elif vaspdir.converged and path == relax_dir:
            inputset = IMDDerivedInputSet(
                name="SCF",
                directory=str(relax_dir),
                user_incar_settings={
                    'ENCUT': 550,
                    'NSW': 0, 'IBRION': -1, 'ISMEAR': -5,
                    # Some runs crash with ALGO = Normal
                    # NCORE = 16 and 8 also sometimes crash
                    'NELM': 200, 'ALGO': 'All', 'NCORE': 4},
                user_kpoints_settings={'grid_density': 10000}
            )
            inputset.write_input(scf_dir)
            print(f"Created new SCF VASP input at {scf_dir}")
            with chdir(scf_dir):
                gorun.run(GORUN_ARGS)
            submitted_jobs = True
            continue
        elif (path / 'RUNNING').is_file() or slurm.directory_queued_p(path):
            print(f"VASP still running in {path}")
            submitted_jobs = True
        elif (path / 'CONTCAR').is_file():
            print(f"VASP unconverged in {path}")
            has_unconverged = True
        else:
            with chdir(path):
                if (path / 'gorun_ready').is_file():
                    print(f'Please submit VASP run in {path}')
                else:
                    gorun.run(GORUN_ARGS)
                submitted_jobs = True
    if submitted_jobs:
        print("Submitted slurm jobs. Waiting for them to finish")
        return 'running'
    if has_unconverged:
        print("No more jobs to run and unconverged jobs found. Check manually. Exiting")
        return 'unconverged'
    return 'converged'


def produce_new_generation(da, mutation_probability):
    """Add new generation to DA.
    """
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    # population_size = int(len(list(da.c.select(generation=0)))/2)
    population_size = 10
    # Load function to measure distance between atoms
    cnt = da.get_slab()
    all_atom_types = get_all_atom_types(cnt, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.8)

    comp = InteratomicDistanceComparator(
        n_top=n_to_optimize,
        pair_cor_cum_diff=0.015,
        pair_cor_max=0.7,
        dE=0.02,
        mic=True,
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

    print(f'Creating new population of size {population_size}')
    size = 0
    for _ in range(population_size):
        a1, a2 = population.get_two_candidates()
        a3, desc = pairing.get_new_individual([a1, a2])
        if a3 is None:
            continue
        # Check if we want to do a mutation
        if random() < mutation_probability:
            # This is necessary because add_unrelaxed_candidate
            # sets parent guid and get_new_individual later fails
            # if we try to mutate something without guid.
            da.add_unrelaxed_candidate(a3, description=desc)
            size += 1
            a3_mut, desc = mutations.get_new_individual([a3])
            if a3_mut is not None:
                da.add_unrelaxed_candidate(a3_mut, desc)
                a3 = a3_mut
        else:
            da.add_unrelaxed_candidate(a3, description=desc)
        size += 1
    if size == 0:
        print("No new offspring can be created. Aborting")
        sys.exit(1)


def run_ga(db_file, reference_vasp, mutation_probability=0.3, max_generations=None):
    """Find ground states for seed structures from DB_FILE using generic algorithm.
    Use REFERENCE_VASP for INCAR setup
    1. Relax unrelaxed seed structures.
    2. Iteratively search GS using already known structures, relaxing one by one.
    """
    # Initialize the different components of the GA
    if not os.path.exists(db_file):
        print("File with seeds and candidates does not exist")
        sys.exit(1)

    shutil.copyfile(db_file, db_file + '.bak')
    da = DataConnection(db_file)
    # Get atomic number of specie to optimize
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)

    def exit_restoring_db():
        shutil.copyfile(db_file + '.bak', db_file)
        sys.exit(1)

    vaspinput = IMDDerivedInputSet(
        directory=reference_vasp,
        # Increase NSW as it is not enough for some CNT
        user_incar_settings={"ISIF": 2, 'IBRION': 2, 'NSW': 1000},
        force_prev_kpoints_file=True,
        )

    while True:
        generation = da.get_generation_number()
        if max_generations and generation > max_generations:
            print(f"Reached maximum number of generations")
            sys.exit(0)
        print(f"Generation {generation}")
        status = relax_all_unrelaxed(da, vaspinput, f"{n_to_optimize}_Na")
        if status == 'unconverged':
            exit_restoring_db()
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
    parser.add_argument("--max_generations", type=int, default=4, help="Maximum number of generations to produce")
    args = parser.parse_args()
    run_ga(args.structure_db, args.cnt_ref, args.mutation_probability, args.max_generations)
