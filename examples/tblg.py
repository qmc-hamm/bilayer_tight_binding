import sys
import json
import ase
from pythtb import *
import matplotlib.pyplot as plt
from bilayer_tight_binding.api import tb_model
# Use our and mk model to compute 9.4 degree twist bands

# Compute hoppings
d = json.load(open('tblg.json'))
lattice_vectors = d['latvec']
atomic_basis = d['atoms']
ase_atoms = ase.Atoms(['C']*len(atomic_basis), positions = atomic_basis, cell = lattice_vectors, pbc = True)
letb = tb_model(ase_atoms, 'letb')
mk = tb_model(ase_atoms, 'mk')

# Compute bands
k=[[1/3., 2/3.], [0.0, 0.0], [0.5, 0.0], [2/3., 1/3.]]
(k_vec, k_dist, k_node) = letb.k_path(k, 100)
evals = letb.solve_all(k_vec)
evals -= min(evals[int(len(atomic_basis)/2), :])

evals_mk = mk.solve_all(k_vec)
evals_mk -= min(evals_mk[int(len(atomic_basis)/2), :])

# plot band structure
fig, ax = plt.subplots(figsize = (3,3))
for i in range(len(atomic_basis)):
    ax.plot(k_dist, evals[i,:], color='k')
    ax.plot(k_dist, evals_mk[i,:], color='r')

# figure formatting
ax.set_ylabel(r'$E - E_F$ (eV)')
ax.set_ylim((-2, 2))
ax.set_xticks(k_node)
ax.set_xticklabels(["K", "$\Gamma$", "M", "$K^\prime$"])
ax.set_xlim(k_node[0], k_node[-1])
fig.savefig("tblg.pdf", bbox_inches='tight')
