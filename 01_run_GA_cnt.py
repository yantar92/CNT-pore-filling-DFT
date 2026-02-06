# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:12:08 2026

@author: thefo
"""

from random import random
import os

# from ase.calculators.emt import EMT
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


# from ase.visualize import view

def check_converged(path):
    '''
    
    Verify that all calculations already converged.

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    files_file = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        
    if 'CONTCAR' not in files_file:
        print(f"Path: {path} does not contains a CONTCAR file, please first run the calculation.")
        sys.exit(1)
    
    else:
        vaspdir = IMDGVaspDir(path)
        if vaspdir.converged == False:
            print(f"Path: {path} the calculation is not converged, check is the calculation still running or have any issue.")
            sys.exit(1)
        else:
            pass

def run_VASP_GA(da, population_size, comp, pairing, mutation_probability, mutations, cwd, number_atoms, ind_cont, encut, grid_density):
    # create the population
    population = Population(
        data_connection=da, population_size=population_size, comparator=comp
    )
    
    for i in range(population_size):
        print(f'Now starting configuration number {i} for n-th gen: {ind_cont/population_size}')
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
        
        # Relax the new candidate
        path_struc_post = cwd + '/' + str(number_atoms).zfill(3) + '_Na/01_Relax_post_seeds/' + str(i + ind_cont).zfill(2)
        if not os.path.exists(path_struc_post):
            os.makedirs(path_struc_post)
        
        os.chdir(path_struc_post)
        
        vasp_inputs(a3, path_struc_post, encut, grid_density)
    
    print("All calculations were submitted.")
    sys.exit(1)
    

def vasp_inputs(struc, path, encut, grid_density):
    
    # 1. Write POSCAR and POTCAR
    write('POSCAR', struc, format='vasp')
    pm_atoms = Structure.from_file('POSCAR')
    
    symbols_struc = struc.get_chemical_symbols()
    
    if 'Na' in symbols_struc:
        potcar = Potcar(
            symbols=['C', 'Na_pv'],
            functional="PBE"
            )
    elif 'Li' in symbols_struc:
        potcar = Potcar(
            symbols=['C', 'Li_sv'],
            functional="PBE"
            )
    else:
        potcar = Potcar(
            symbols=['C'],
            functional="PBE"
            )
    potcar.write_file(os.path.join(path, "POTCAR"))
    
    # 2. Write INCAR
    incar_content = generate_incar(encut)
    with open(os.path.join(path, "INCAR"), "w", encoding="utf-8") as f:
        f.write(incar_content)
    
    # 3. Write KPOINTS
    kpoints = Kpoints.automatic_density(pm_atoms, grid_density, force_gamma=True)
    with open(os.path.join(path, "KPOINTS"), "w", encoding="utf-8") as f:
        f.write(str(kpoints))
    
    # submit = generate_submit(vasp_cmd)
    # with open(os.path.join(path, "sub"), "w", encoding="utf-8") as f:
    #     f.write(submit)
    
    subprocess.Popen(
        "gorun 1 24:00:00 >> log",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detaches from your Python process session
        )
    
    time.sleep(30)

def generate_incar(encut):
    return f"""SYSTEM = SCF Calculation (PBE-D3(BJ)/Becke-Johnson)
ENCUT = {encut}
PREC = Normal
NCORE = 4
EDIFF = 1E-6
NSW = 1000
IBRION = 2
ISIF = 2
ISMEAR = 0
SIGMA = 0.04
ISPIN = 1
GGA = PE
ISTART = 0
ICHARG = 2
LCHARG = .FALSE.
LWAVE = .FALSE.
LREAL = Auto
NELM = 200
"""

# def generate_submit(command_vasp):
#     return f"""#!/bin/bash -l
# #SBATCH --job-name=test_run
# #SBATCH -N 1
# #SBATCH -t 24:00:00
# #SBATCH --ntasks-per-node=96
# #SBATCH --partition=plgrid
# module load GCC/13.2.0 OpenMPI/5.0.3 OpenBLAS/0.3.24 ScaLAPACK/2.2.0-fb FFTW/3.3.10

# {command_vasp} >> log_relax
# """

# vasp_cmd = os.environ["ASE_VASP_COMMAND"]

def run_ga(number_atoms, population_size, n_gen, encut, grid_density):
    
    # Change the following parameter to suit your needs
    mutation_probability = 0.3
    
    ### Get current path
    cwd = os.getcwd()
    
    # Initialize the different components of the GA
    db_file = 'GA_' + str(number_atoms) + '_Na_CNT_15_0.db'
    if not os.path.exists(db_file):
        print("File with seeds and candidates does not exist")
        sys.exit(1)
    
    shutil.copyfile(db_file, 'back_up.db')
    da = DataConnection(db_file)
    
    # Get atomic number of specie to optimize
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    
    len_relax_struc = len(da.get_all_relaxed_candidates())
    
    # Load function to measure distance between atoms
    cnt = da.get_slab()
    all_atom_types = get_all_atom_types(cnt, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.8)
    
    n_to_optimize = len(atom_numbers_to_optimize)
    comp = InteratomicDistanceComparator(
        n_top=n_to_optimize,
        pair_cor_cum_diff=0.015,
        pair_cor_max=0.7,
        dE=0.02,
        mic=False,
    )
    
    ## Operations that will be performed for generate mutations
    
    pairing = CutAndSplicePairing(cnt, n_to_optimize, blmin)
    
    if number_atoms == 1:
    
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
    
    path_init_seeds = cwd + '/' + str(number_atoms).zfill(3) + '_Na/00_Relax_seeds'
        
    if not os.path.exists(path_init_seeds):
        for i in range(population_size):
            a = da.get_all_unrelaxed_candidates()[i]

            path_struc = cwd + '/' + str(number_atoms).zfill(3) + '_Na/00_Relax_seeds/' + str(i).zfill(2)
            if not os.path.exists(path_struc):
                os.makedirs(path_struc)
            
            os.chdir(path_struc)
            vasp_inputs(a, path_struc, encut, grid_density)
        
        print("All calculations were submitted.")
        sys.exit(1)
    else:
        if len_relax_struc == 0:
            for i in range(population_size):
                ## First check if all the outcar files exist
                path_struc = cwd + '/' + str(number_atoms).zfill(3) + '_Na/00_Relax_seeds/' + str(i).zfill(2)
                os.chdir(path_struc)
                check_converged(path_struc)
            
            for i in range(population_size):
                ## If all the outcar files exist then if read all the energies
                path_struc = cwd + '/' + str(number_atoms).zfill(3) + '_Na/00_Relax_seeds/' + str(i).zfill(2)
                os.chdir(path_struc)
                
                try:
                    a = da.get_all_unrelaxed_candidates()[i]
                    
                    # Read from OUTCAR
                    a_tem = read('OUTCAR', index=-1)                    
                    energy = a_tem.get_potential_energy()
                    forces = a_tem.get_forces()
                    
                    # Set positions, energy, forces, and stress tensor to the structure
                    a.set_positions(a_tem.get_positions())                    
                    a.calc = SinglePointCalculator(a, energy=energy, forces=forces)
                    print(f"It was stored the energy and forces into the object for ith: {i}")
                    a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
                    da.add_relaxed_step(a)
                    
                    del a
                    
                except Exception as e:
                    print(f"It was no possible to read OUTCAR error: {e} for i-th calculation: {i}")
                    shutil.copyfile(cwd + '/' + 'back_up.db', cwd + '/' + db_file)
                    sys.exit(1)
            
            run_VASP_GA(da, population_size, comp, pairing, mutation_probability, mutations, cwd, number_atoms, 0, encut, grid_density)
            
        else:
            cont = (len_relax_struc // population_size) - 1
            
            for i in range(population_size):
                ## First check if all the outcar files exist
                path_struc_post = cwd + '/' + str(number_atoms).zfill(3) + '_Na/01_Relax_post_seeds/' + str(i + population_size*cont).zfill(2)
                os.chdir(path_struc_post)
                check_converged(path_struc_post)
            
            for i in range(population_size):
                ## If all the outcar files exist then if read all the energies
                path_struc_post = cwd + '/' + str(number_atoms).zfill(3) + '_Na/01_Relax_post_seeds/' + str(i + population_size*cont).zfill(2)
                os.chdir(path_struc_post)
                
                try:
                    a = da.get_all_unrelaxed_candidates()[i]
                    
                    # Read from OUTCAR
                    a_tem = read('OUTCAR', index=-1)                    
                    energy = a_tem.get_potential_energy()
                    forces = a_tem.get_forces()
                    
                    # Set positions, energy, forces, and stress tensor to the structure
                    a.set_positions(a_tem.get_positions())                    
                    a.calc = SinglePointCalculator(a, energy=energy, forces=forces)
                    print(f"It was stored the energy and forces into the object for ith: {i}")
                    a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
                    da.add_relaxed_step(a)
                    
                    del a
                except Exception as e:
                    print(f"It was no possible to read OUTCAR error: {e} for i-th calculation: {i + population_size*cont}")
                    shutil.copyfile(cwd + '/' + 'back_up.db', cwd + '/' + db_file)
                    sys.exit(1)
            
            if (cont + 1) <= n_gen:
                run_VASP_GA(da, population_size, comp, pairing, mutation_probability, mutations, cwd, number_atoms, population_size*(cont + 1), encut, grid_density)
            
            else:
                print("All calculations are completed.")
                sys.exit(1)

parser = argparse.ArgumentParser(description="Script to optimize Na into optimized CNT.")
parser.add_argument("--number_atoms", type=int, required=True, help="Number of Na")
parser.add_argument("--size_seeds", type=int, default=20, help="Number of seeds")
parser.add_argument("--encut", type=float, default=550, help="ENCUT")
parser.add_argument("--grid_density", type=int, default=6000, help="Density for generating k-mesh grid")
parser.add_argument("--n_gen", type=int, default=3, help="Number of generation")

args = parser.parse_args()                

# number_Na = 1
# encut = 550
# grid_density = 6000
# population_size = 20
# n_gen = 5

run_ga(args.number_atoms, args.size_seeds, args.n_gen, args.encut, args.grid_density)

# da = run_ga(1, 20, 5, 550, 6000)


#db_file = 'GA_1_Na_CNT_15_0.db'
#da = DataConnection(db_file)
