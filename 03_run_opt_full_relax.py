"""
Read energies from relax.2.optB88-vdW and fully
relax CNT lengths.
"""
from pathlib import Path
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir

INCAR_PY = """
import sys
import numpy as np
from ase.eos import EquationOfState
from ase.io import read
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir


SCAN_THRESHOLD = 0.04 # scan threshold

# 1. Create your configurations (looping over z-lengths)
# Assume 'atoms' is your starting structure
z_original = atoms.cell[2, 2]
# !! 5 POINTS!
z_factors = np.linspace(1 - SCAN_THRESHOLD, 1 + SCAN_THRESHOLD, 5)
energies = []
volumes = []

for f in z_factors:
    atoms.cell[2, 2] = z_original * f
    # Important: scale_atoms=True moves atoms proportionally in z
    atoms.set_cell(atoms.cell, scale_atoms=True)
    energies.append(atoms.get_potential_energy())
    volumes.append(atoms.get_volume())
    for f, energy in zip(z_factors, energies):
        print(f"{z_original * f} {energy}")
    d = IMDGVaspDir('.')
    if not (d.converged_electronic and d.converged_ionic):
         print('VASP not converged. Aborting')
         sys.exit(1)

# 2. Fit the data
# Even though we varied Z, we fit Energy vs. Volume
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()

# 3. Calculate your optimal Z from the optimal Volume
opt_z = v0 / (atoms.cell[0,0] * atoms.cell[1,1] * np.sin(np.deg2rad(atoms.cell.cellpar()[5])))

atoms.cell[2, 2] = opt_z
# Important: scale_atoms=True moves atoms proportionally in z
atoms.set_cell(atoms.cell, scale_atoms=True)
energies.append(atoms.get_potential_energy())
print(f"Optimal z-length: {opt_z}, E = {energies[-1]}")
for f, energy in zip(z_factors, energies):
    print(f"{z_original * f} {energy}")
print(f"{opt_z} {energies[-1]}")
"""

vaspdirs = IMDGVaspDir.read_vaspdirs(
    '.', path_filter=lambda d: d.name == 'relax.2.optB88-vdW')
for p in vaspdirs:
    print(p)
    vaspdir = vaspdirs[p]
    if not vaspdir.converged:
        print(f"Skipping unconverged dir {p}")
        continue
    target_dir = Path(p).parent / "relax.final.optB88-vdW"
    if target_dir.is_dir():
        print(f"Already present {target_dir}. Skipping")
        continue
    inputset = IMDDerivedInputSet(directory=vaspdir)
    inputset.write_input(target_dir)
    with open(target_dir / "INCAR.py", "w") as f:
        f.write(INCAR_PY)
    print(f'Wrote to {target_dir}')
