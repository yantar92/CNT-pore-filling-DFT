"""
Read energies from formation_en.txt (PBE!) and re-relax CNTs with
lowest energies and +XmeV above.
Unlike PBE case, I optimize further:
lower ENERGY_THRESHOLD and only 5 points for length optimization.
"""
from pathlib import Path
import pandas as pd
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet
from IMDgroup.pymatgen.core.structure import structure_matches

ENERGY_THRESHOLD = 5  # meV/atom

df = pd.read_csv('formation_en.txt', sep=' ')
min_energies = df.groupby('Formula')['Formation Energy (meV/atom)'].transform('min')
final_result = df[df['Formation Energy (meV/atom)'] <= (min_energies + ENERGY_THRESHOLD)].copy()

known_structures = []
for p in sorted(final_result['ID']):
    if 'gen' not in p:
        continue
    if (Path(p).parent.parent / "FULL_RELAX_GENERATED").is_file():
        print(f"Skipping because of {(Path(p).parent.parent / 'FULL_RELAX_GENERATED')}")
        continue
    print(p)
    target_dir = Path(p).parent / "relax.2.optB88-vdW"
    inputset = IMDDerivedInputSet(
        directory=p, functional='optB88-vdW',
        user_incar_settings={'ALGO': 'All', 'PREC': 'Accurate', 'NELM': 200, 'NELMIN': 6})
    print("Read VASP output")
    if not structure_matches(inputset.structure, known_structures, multithread=True):
        known_structures.append(inputset.structure.copy())
        if target_dir.is_dir():
            print(f"Already present {target_dir}. Skipping")
            continue
        for site in inputset.structure:
            del site.properties['selective_dynamics']
        inputset.write_input(target_dir)
        # with open(target_dir / "INCAR.py", "w") as f:
        #     f.write(INCAR_PY)
        print(f'Wrote to {target_dir}')
    else:
        print('Skipping known structure')
