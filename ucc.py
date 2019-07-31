'''
Computing the UCC(4) correlation energy using an RHF reference
Spin orbital equations
Reference used:
    - Bartlett:1988:133
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
from helper_cc import *
import time

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)
numpy_memory = 2

# Set Psi4 options
mol = psi4.geometry("""           
0 1
O   0.00000000     0.00000000    -0.12464853
H   0.00000000    -1.43820727     0.98913129
H   0.00000000     1.43820727     0.98913129

noreorient
symmetry c1
units bohr                                      
""")

psi4.set_options({'basis': '3-21G', 'scf_type': 'pk',
                  'freeze_core': 'false', 'e_convergence': 1e-12,
                  'd_convergence': 1e-12, 'save_jk': 'true'})

# Set for CCSD
E_conv = 1e-8
R_conv = 1e-7
maxiter = 40
compare_psi4 = False

# Compute RHF energy with psi4
psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-12})
psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-12})
e_scf, wfn = psi4.energy('SCF', return_wfn=True)
print('SCF energy: {}\n'.format(e_scf))
print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

# Create Helper_CCenergy object
hucc = HelperUCC4(wfn) 

ucc_e = hucc.do_CC(e_conv=1e-10, r_conv =1e-10, maxiter=40)

print('UCC(4) Correlation Energy: {}'.format(ucc_e))
print('UCC(4) Energy: {}'.format(e_scf+ucc_e))
