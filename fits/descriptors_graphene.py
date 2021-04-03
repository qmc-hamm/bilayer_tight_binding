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

def ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """ 
    Converts displacement indices to 
    physical distances and all the 2-body terms we care about
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    dxy - Distance in Bohr, projected in the x/y plane
    dz  - Distance in Bohr, projected onto the z axis
    """
    displacement_vector = di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]

    displacement_vector_xy = displacement_vector[:, :2] 
    displacement_vector_z =  displacement_vector[:, -1] 

    dxy = np.linalg.norm(displacement_vector_xy, axis = 1)
    dz = np.abs(displacement_vector_z)
    return dxy, dz
