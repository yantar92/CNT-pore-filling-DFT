# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 08:59:18 2026

Create a CNT structure and run relaxation.

@author: thefo
"""

import os
import subprocess
import argparse
from contextlib import chdir
import numpy as np
from ase.build import nanotube
from pymatgen.io.ase import AseAtomsAdaptor
from IMDgroup.pymatgen.io.vasp.sets import IMDStandardVaspInputSet_relax
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.gorun import gorun


def main(n, m, length, vacuum=20, defect=None, encut=500, density=6000):
    """Build VASP input for CNT and run VASP.
    CNT has n,m chirality, length, and adds vacuum space around.
    """
    # Modify the quirality to create a new CNT
    cnt_struct = AseAtomsAdaptor.get_structure(
        nanotube(n, m, length=length, vacuum=vacuum))

    if defect is None:
        pass
    elif defect == 'MV':
        # Remove one carbon at random
        cnt_struct.remove_sites([np.random.randint(0, len(cnt_struct))])
    elif defect == 'DW':
        raise NotImplementedError
    elif defect == 'SW':
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown defect type: {defect}')

    vasp_input = IMDStandardVaspInputSet_relax(
        name=f'CNT_{n},{m}_{length}_{vacuum}',
        functional='pbe',
        structure=cnt_struct,
        user_incar_settings={
            'ENCUT': encut,
            'ISIF': Incar.ISIF_RELAX_POS,
            'IBRION': Incar.IBRION_IONIC_RELAX_CGA},
        user_kpoints_settings={'grid_density': density},
    )

    vasp_input.write_input(output_dir=vasp_input.name)

    with chdir(vasp_input.name):
        gorun.run(argparse.Namespace(
            number_of_nodes="1",
            time_limit="24:00:00",
            mark=True,
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CNT and run relaxation.")
    parser.add_argument("n", type=int, help="(n, m) CNT")
    parser.add_argument("m", type=int, help="(n, m) CNT")
    parser.add_argument("length", type=int, help="CNT length (repetitions)")
    parser.add_argument("--vacuum", type=float, default=20, help="CNT vacuum around (default: 20A)")
    parser.add_argument("--defect", type=str, default=None, help="Defect to introduce (MV, DV, SW)")
    parser.add_argument("--encut", type=float, default=500, help="ENCUT (default: 500eV)")
    parser.add_argument("--kpoints", type=float, default=6000, help="Kpoint density (default: 6000)")
    args = parser.parse_args()
    main(args.n, args.m, args.length, args.vacuum, args.defect, args.encut, args.kpoints)
