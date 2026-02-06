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
from ase.build import nanotube
from pymatgen.io.ase import AseAtomsAdaptor
from IMDgroup.pymatgen.io.vasp.sets import IMDStandardVaspInputSet_relax
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.gorun import gorun


def main(n, m, length, vacuum=20):
    """Build VASP input for CNT and run VASP.
    CNT has n,m chirality, length, and adds vacuum space around.
    """
    # Modify the quirality to create a new CNT
    struc_base_sc = nanotube(n, m, length=length, vacuum=vacuum)

    vasp_input = IMDStandardVaspInputSet_relax(
        name=f'CNT_{n},{m}_{length}_{vacuum}',
        functional='pbe',
        structure=AseAtomsAdaptor.get_structure(struc_base_sc),
        user_incar_settings={
            'ENCUT': 550,
            'ISIF': Incar.ISIF_FIX_NONE,
            'IBRION': Incar.IBRION_IONIC_RELAX_CGA},
        user_kpoints_settings={'grid_density': 6000},
    )

    vasp_input.write_input(output_dir=vasp_input.name)

    with chdir(vasp_input.name):
        gorun.run(argparse.Namespace(
            number_of_nodes="1",
            time_limit="24:00:00",
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CNT and run relaxation.")
    parser.add_argument("n", type=int, help="(n, m) CNT")
    parser.add_argument("m", type=int, help="(n, m) CNT")
    parser.add_argument("length", type=float, help="CNT length")
    parser.add_argument("--vacuum", type=float, default=20, help="CNT vacuum around (default: 20A)")
    args = parser.parse_args()
    main(args.n, args.m, args.length, args.vacuum)
