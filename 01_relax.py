# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 08:59:18 2026

@author: thefo
"""

# from ase.io import read, write
from ase.build import nanotube
from ase.visualize import view

import datetime
import os
import sys
import time

# import numpy as np

from ase.io import read, write
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Potcar
from pymatgen.io.vasp.inputs import Kpoints
from ase.build.supercells import make_supercell

# import shlex
import subprocess

def generate_incar_pbe_d3_bj(encut):
    return f"""SYSTEM = SCF Calculation (PBE-D3(BJ)/Becke-Johnson)
ENCUT = {encut}
PREC = Normal
NCORE = 4
EDIFF = 1E-6
NSW = 1000
IBRION = 2
ISIF = 3
ISMEAR = 0
SIGMA = 0.04
ISPIN = 1
GGA = PE
IVDW = 0
ISTART = 0
ICHARG = 2
LCHARG = .FALSE.
LWAVE = .FALSE.
LREAL = Auto
NELM = 200
"""

#def generate_submit(command_vasp):
#    return f"""#!/bin/bash -l
##SBATCH --job-name=test_run
##SBATCH -N 1
##SBATCH -t 24:00:00
##SBATCH --ntasks-per-node=96
##SBATCH --partition=plgrid
#module load GCC/13.2.0 OpenMPI/5.0.3 OpenBLAS/0.3.24 ScaLAPACK/2.2.0-fb FFTW/3.3.10
#
#{command_vasp} >> log_relax
#"""
#
## ----------------------------------------------------------------------
#
#vasp_cmd = os.environ["ASE_VASP_COMMAND"]

encut = 550
grid_density = 6000

cwd = os.getcwd()

### Modify the quirality to create a new CNT
struc_base_sc = nanotube(15, 0, length=1, vacuum=20)

path_struc = cwd

os.chdir(path_struc)

write('POSCAR', struc_base_sc, format='vasp')
pm_atoms = Structure.from_file('POSCAR')
poscar = Poscar(pm_atoms)
potcar = Potcar(
    symbols=poscar.site_symbols,
    functional="PBE"
    )
potcar.write_file(os.path.join(path_struc, "POTCAR"))


# 2. Write INCAR using the current functional
incar_content = generate_incar_pbe_d3_bj(encut)
with open(os.path.join(path_struc, "INCAR"), "w", encoding="utf-8") as f:
    f.write(incar_content)

# 3. Write KPOINTS
kpoints = Kpoints.automatic_density(pm_atoms, grid_density, force_gamma=True)
with open(os.path.join(path_struc, "KPOINTS"), "w", encoding="utf-8") as f:
    f.write(str(kpoints))

# 4. Run calculations, it is sequential

#submit = generate_submit(vasp_cmd)
#with open(os.path.join(path_struc, "submit.sh"), "w", encoding="utf-8") as f:
#    f.write(submit)

subprocess.Popen(
    ["gorun 1 24:00:00 >> log"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,  # detaches from your Python process session
    )
