import sys
sys.path.append('../')
from model import graphene 
from pythtb import *
import matplotlib.pyplot as plt

# Use our model to compute hoppings
R0 = 2.6834 # Bohr
lattice_vectors = np.array([[np.sqrt(3) * R0, 0.0, 0.0], [-np.sqrt(3)/2 * R0, 3/2 * R0, 0.0]])
atomic_basis    = np.array([[0, 0, 0], [0, R0, 0]])
i =  [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,  1]
j =  [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,  0]
di = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, -1]
dj = [0, 1, 1, 0, 1, 1, 0, 1, 1, 2, 0,  0]
hoppings = graphene(lattice_vectors, atomic_basis, i, j, di, dj)

# compute and plot the band structure
z = 0
lat = lattice_vectors[:, :-1]
orb = atomic_basis[:, :-1]
gra = tb_model(2, 2, lat, orb)
k=[[0.0, 0.0], [2/3., -1./3.], [0.5, 0]]

color = {3:'k', 9:'b', 12:'m'}
label = {3:'t01', 9:'t01, t02', 12:'t01, t02, t03'}
fig, ax = plt.subplots()
for ii, jj, dii, djj, hopping in zip(i, j, di, dj, hoppings):
    z += 1
    gra.set_hop(hopping, ii, jj, [dii, djj])
    if (z==3) or (z==9) or (z==12):
        (k_vec, k_dist, k_node) = gra.k_path(k, 100)
        evals = gra.solve_all(k_vec)
        evals -= min(evals[1, :])
        ax.plot(k_dist, evals[0,:], color=color[z])
        ax.plot(k_dist, evals[1,:], color=color[z], label=label[z])
ax.set_xticks(k_node)
ax.set_xticklabels(["$\Gamma$", "K", "M"])
ax.set_xlim(k_node[0], k_node[-1])
ax.legend(loc='best')
fig.savefig("graphene.pdf", bbox_inches='tight')
