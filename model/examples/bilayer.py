import sys
import json
sys.path.append('../')
from model import bilayer
from pythtb import *
import matplotlib.pyplot as plt

def compute_bands(lattice_vectors, atomic_basis, i, j, di, dj, hoppings):
    lat = lattice_vectors[:, :-1]
    orb = atomic_basis[:, :-1]
    gra = tb_model(2, 2, lat, orb)
    k=[[0.0, 0.0], [2/3., -1./3.], [0.5, 0.]]
    for ii, jj, dii, djj, hopping in zip(i, j, di, dj, hoppings):
        gra.set_hop(hopping, ii, jj, [dii, djj])
    (k_vec, k_dist, k_node) = gra.k_path(k, 100)
    evals = gra.solve_all(k_vec)
    evals -= min(evals[2, :])
    return evals, k_dist, k_node

# use model to compute hoppings
R0 = 2.6834 # Bohr
lattice_vectors = np.array([[np.sqrt(3) * R0, 0.0, 0.0], [-np.sqrt(3)/2 * R0, 3/2 * R0, 0.0]])

# aa model - 1NN in plane, 1 + 2NN out of plane
i =  [0, 1, 1, 2, 3, 3, 0, 1, 1, 1, 1, 3, 3, 3]
j =  [1, 0, 0, 3, 2, 2, 2, 3, 2, 2, 2, 0, 0, 0]
di = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
dj = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
atomic_basis_aa = np.array([[0, 0, 0], [0, R0, 0], [0, 0, 6.50], [0, R0, 6.50]])
hoppings_aa = bilayer(lattice_vectors, atomic_basis_aa, i, j, di, dj)
evals_aa, k_dist, k_node = compute_bands(lattice_vectors, atomic_basis_aa, i, j, di, dj, hoppings_aa)

#atomic_basis_ab = np.array([[0, 0, 0], [0, R0, 0], [0, R0, 6.50], [0, 2 * R0, 6.50]])
#hoppings_ab = bilayer(lattice_vectors, atomic_basis, i, j, di, dj)

#atomic_basis_sp = np.array([[0, 0, 0], [0, R0, 0], [0, 1.5 * R0, 6.50], [0, 2.5 * R0, 6.50]])
#hoppings_sp = bilayer(lattice_vectors, atomic_basis, i, j, di, dj)

# compute the band structure
# plot the band structure
fig, ax = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True, figsize = (11, 3))
for i in range(len(evals_aa)):
    ax[0].plot(k_dist, evals_aa[i,:], 'k-')
ax[0].set_ylabel(r'$E - E_F$ (eV)')
ax[0].set_ylim((-10, 15))
ax[0].set_xticks(k_node)
ax[0].set_xticklabels(["$\Gamma$", "K", "M"])
ax[0].set_xlim(k_node[0], k_node[-1])
fig.savefig("bilayer.pdf", bbox_inches='tight')
