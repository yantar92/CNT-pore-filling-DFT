"""
Read energies from formation_en.txt and fully
relax CNTs with lowest energies and +XmeV above.
"""
from pathlib import Path
import pandas as pd
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet
from IMDgroup.pymatgen.core.structure import structure_matches

ENERGY_THRESHOLD = 10  # meV/atom

INCAR_PY = """
import numpy as np
from ase.eos import EquationOfState
from ase.io import read


SCAN_THRESHOLD = 0.04 # scan threshold

# 1. Create your configurations (looping over z-lengths)
# Assume 'atoms' is your starting structure
z_original = atoms.cell[2, 2]
z_factors = np.linspace(1 - SCAN_THRESHOLD, 1 + SCAN_THRESHOLD, 7) # 7 points is usually enough
energies = []
volumes = []

for f in z_factors:
    atoms.cell[2, 2] = z_original * f
    # Important: scale_atoms=True moves atoms proportionally in z
    atoms.set_cell(atoms.cell, scale_atoms=True)
    energies.append(atoms.get_potential_energy())
    volumes.append(atoms.get_volume())

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

df = pd.read_csv('formation_en.txt', sep=' ')
min_energies = df.groupby('Formula')['Formation Energy (meV/atom)'].transform('min')
final_result = df[df['Formation Energy (meV/atom)'] <= (min_energies + ENERGY_THRESHOLD)].copy()

known_structures = []
for p in sorted(final_result['ID']):
    if 'gen' not in p:
        continue
    if (Path(p) / "FULL_RELAX_GENERATED").is_file():
        print(f"Skipping because of {(Path(p) / 'FULL_RELAX_GENERATED')}")
        continue
    print(p)
    target_dir = Path(p).parent / "relax.final"
    inputset = IMDDerivedInputSet(directory=p)
    print("Read VASP output")
    if not structure_matches(inputset.structure, known_structures, multithread=True):
        known_structures.append(inputset.structure.copy())
        if target_dir.is_dir():
            print(f"Already present {target_dir}. Skipping")
            continue
        for site in inputset.structure:
            del site.properties['selective_dynamics']
        inputset.write_input(target_dir)
        with open(target_dir / "INCAR.py", "w") as f:
            f.write(INCAR_PY)
        print(f'Wrote to {target_dir}')
    else:
        print('Skipping known structure')
