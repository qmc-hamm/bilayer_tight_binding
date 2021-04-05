import sys
import json
from pythtb import *
import matplotlib.pyplot as plt
from bilayer_tight_binding.model import graphene 

# Use our model to compute hoppings up to 3NN
R0 = 2.6834 # Bohr
lattice_vectors = np.array([[np.sqrt(3) * R0, 0.0, 0.0], [-np.sqrt(3)/2 * R0, 3/2 * R0, 0.0]])
atomic_basis    = np.array([[0, 0, 0], [0, R0, 0]])
i =  [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,  1]
j =  [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,  0]
di = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, -1]
dj = [0, 1, 1, 0, 1, 1, 0, 1, 1, 2, 0,  0]
hoppings = graphene(lattice_vectors, atomic_basis, i, j, di, dj)

# compute the band structure
lat = lattice_vectors[:, :-1]
orb = atomic_basis[:, :-1]
gra = tb_model(2, 2, lat, orb)
k=[[0.0, 0.0], [2/3., -1./3.], [1./3., 1./3.]]
for ii, jj, dii, djj, hopping in zip(i, j, di, dj, hoppings):
    gra.set_hop(hopping, ii, jj, [dii, djj])
(k_vec, k_dist, k_node) = gra.k_path(k, 100)
evals = gra.solve_all(k_vec)
evals -= min(evals[1, :])

# plot band structure
fig, ax = plt.subplots(figsize = (3,3))
ax.plot(k_dist, evals[0,:], color='b', lw = 3, alpha = 0.5)
ax.plot(k_dist, evals[1,:], color='b', lw = 3, alpha = 0.5)

# plot the reference data
ref = json.load(open('graphene.json','r'))
ref['path'] = np.array(ref['path'])
ref['bands'] = np.array(ref['bands'])
ref['path'] /= max(ref['path']) / max(k_dist)
for i in range(len(ref['bands'])):
    ax.plot(ref['path'], ref['bands'][i], 'k-')

# figure formatting
ax.set_ylabel(r'$E - E_F$ (eV)')
ax.set_ylim((-10, 15))
ax.set_xticks(k_node)
ax.set_xticklabels(["$\Gamma$", "K", "$K^\prime$"])
ax.set_xlim(k_node[0], k_node[-1])
ax.legend(loc='best')
fig.savefig("graphene.pdf", bbox_inches='tight')
