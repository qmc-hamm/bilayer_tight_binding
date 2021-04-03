import h5py
import numpy as np 
import pandas as pd

def nnmat(lattice_vectors, atomic_basis):
    """
    Build matrix which tells you relative coordinates
    of nearest neighbors to an atom i in the supercell

    Returns: nnmat [natom x 3 x 3]
    """
    import scipy.spatial as spatial
    nnmat = np.zeros((len(atomic_basis), 3, 3))

    # Extend atom list
    atoms = []
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            displaced_atoms = atomic_basis + lattice_vectors[np.newaxis, 0] * i + lattice_vectors[np.newaxis, 1] * j
            atoms += [list(x) for x in displaced_atoms]
    atoms = np.array(atoms)

    # Pairwise distance matrix
    distances = spatial.distance.pdist(atoms)
    distances = spatial.distance.squareform(distances)

    # Trim
    distances = distances[:atomic_basis.shape[0]]

    # Loop
    for i in range(distances.shape[0]):
        ind = np.argsort(distances[i])
        nnmat[i] = atoms[ind[1:4]] - atomic_basis[i]

    return nnmat
