import sys
import json
sys.path.append('../')
from model import graphene 
from pythtb import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compute_bands(lattice_vectors, atomic_basis, i, j, di, dj):
    hoppings = graphene(lattice_vectors, atomic_basis, i, j, di, dj)
    lat = lattice_vectors[:, :-1]
    orb = atomic_basis[:, :-1]
    gra = tb_model(2, 2, lat, orb)

    G = [0, 0]
    M = [0.5, 0]
    R = [1./3., 1./3.]
    S = [0, 0.5]
    K = [2 * R[0], -R[1]]
    k=[R, G, K, R, S, G, M]

    for ii, jj, dii, djj, hopping in zip(i, j, di, dj, hoppings):
        gra.set_hop(hopping, ii, jj, [dii, djj])
    (k_vec, k_dist, k_node) = gra.k_path(k, 500)
    evals = gra.solve_all(k_vec)
    evals -= min(evals[1, :])
    return evals, k_dist, k_node

# Strain geometry
R0 = 2.6834 # Bohr
i =  [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,  1]
j =  [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,  0]
di = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, -1]
dj = [0, 1, 1, 0, 1, 1, 0, 1, 1, 2, 0,  0]

lattice_vectors0 = np.array([[np.sqrt(3) * R0, 0.0, 0.0], [-np.sqrt(3)/2 * R0, 3/2 * R0, 0.0]])
atomic_basis0    = np.array([[0, 0, 0], [0, R0, 0]])

# Armchair strain
fig, ax = plt.subplots(figsize = (3,3))
cmap   = mpl.cm.coolwarm
norm   = mpl.colors.Normalize(-2, 2)
for s, strain in enumerate(np.linspace(-0.02, 0.02, 11)):
    lattice_vectors = np.copy(lattice_vectors0)
    atomic_basis = np.copy(atomic_basis0)

    lattice_vectors[:, 1] *= (1 + strain)
    atomic_basis[:, 1] *=    (1 + strain)
    evals, k_dist, k_node = compute_bands(lattice_vectors, atomic_basis, i, j, di, dj)

    ax.plot(k_dist, evals[0,:], color=cmap(norm(strain * 100)), lw = 1)
    ax.plot(k_dist, evals[1,:], color=cmap(norm(strain * 100)), lw = 1)

# figure formatting
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar   = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
cbar.set_label('Strain (%)')
ax.set_ylabel(r'$E - E_F$ (eV)')
ax.set_ylim((-10, 15))
ax.set_xticks(k_node)
ax.set_xticklabels(['R','G','K','R','S','G','M'])
ax.set_xlim(k_node[0], k_node[-1])
ax.legend(loc='best')
fig.savefig("graphene_strain.pdf", bbox_inches='tight')
