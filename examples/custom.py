# Custom usage of the letb, look at tblg.py for default behavior
import json
from pythtb import *
import matplotlib.pyplot as plt
from bilayer_letb.model import letb

# use model to compute hoppings
R0 = 2.6834 # Bohr
lattice_vectors = np.array([[np.sqrt(3) * R0, 0.0, 0.0], [-np.sqrt(3)/2 * R0, 3/2 * R0, 0.0]])
atomic_basis = np.array([[0, 0, 0], [0, R0, 0], [0, 0, 6.646], [0, R0, 6.646]])
i =  [1, 1, 1, 3, 3, 3, 2, 3, 3, 3, 1, 1, 1, 1]
j =  [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 3, 2, 2, 2]
di = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
dj = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
hoppings = letb(lattice_vectors, atomic_basis, i, j, di, dj)

# compute bands
lat = lattice_vectors[:, :-1]
orb = atomic_basis[:, :-1]
gra = tb_model(2, 2, lat, orb)
k=[[0.5, 0.0], [2/3., -1./3.], [0.5, 0.]]
for ii, jj, dii, djj, hopping in zip(i, j, di, dj, hoppings):
    gra.set_hop(hopping, ii, jj, [dii, djj])
(k_vec, k_dist, k_node) = gra.k_path(k, 1000)
evals = gra.solve_all(k_vec)
evals -= min(evals[2, :])

# plot the band structure
fig, ax = plt.subplots(figsize = (3, 3))
for i in range(len(evals)):
    ax.plot(k_dist, evals[i,:], 'k-')
ax.set_ylabel(r'$E - E_F$ (eV)')
ax.set_ylim((-5, 5))
ax.set_xticks(k_node)
ax.set_xticklabels(["M", "K", "M"])
ax.set_xlim(k_node[0], k_node[-1])
fig.savefig("custom.pdf", bbox_inches='tight')
