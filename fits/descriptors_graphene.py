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

def partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Given displacement indices and geometry,
    get indices for partitioning the data

    ###########################################################
    YOU MAY GENERALIZE THIS IN THE FUTURE
    ###########################################################
    """
    distances = ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj)
    ix = np.argsort(distances)
    t01_ix = ix[:3 * len(di) // 39]
    t02_ix = ix[3 * len(di) // 39 :9 * len(di) // 39]
    t03_ix = ix[9 * len(di) // 39 :12 * len(di) // 39]
    return t01_ix, t02_ix, t03_ix

def triangle_height(a, base):
    """
    Give area of a triangle given two displacement vectors for 2 sides
    """
    area = np.linalg.det(
            np.array([a, base, [1, 1, 1]])
    )
    area = np.abs(area)/2
    height = 2 * area / np.linalg.norm(base)
    return height

def t01_descriptors(lattice_vectors, atomic_basis, nnmat, tij, di, dj, ai, aj, partition, typ, uuid, energy):
    # Compute NN distances
    r = d1[:, np.newaxis] * lattice_vectors[0] + d2[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[a2] - atomic_basis[a1] # Relative coordinates
    a = np.linalg.norm(r, axis = 1)
    return {'t': list(tij), 'a': list(a)}

def t02_descriptors(lattice_vectors, atomic_basis, nnmat, tij, di, dj, ai, aj, partition, typ, uuid, energy):
    # Compute NNN distances
    r = d1[:, np.newaxis] * lattice_vectors[0] + d2[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[a2] - atomic_basis[a1] # Relative coordinates
    b = np.linalg.norm(r, axis = 1)

    # Compute h1, h2
    h1 = []
    h2 = []
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in range(len(r)):
        nn = nnmat[a2[i]] + r[i]
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        h1.append(triangle_height(nn[ind[0]], r[i]))
        h2.append(triangle_height(nn[ind[1]], r[i]))
    return {'t': list(tij), 'b': list(b), 'h1': list(h1), 'h2': list(h2)}
    

def t03_descriptors(lattice_vectors, atomic_basis, tij, di, dj, ai, aj):
    """
    Compute t03 descriptors
    """
    # Compute NNNN distances
    r = d1[:, np.newaxis] * lattice_vectors[0] + d2[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[a2] - atomic_basis[a1] # Relative coordinates
    c = np.linalg.norm(r, axis = 1)

    # All other hexagon descriptors
    l = []
    h = []
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in range(len(r)):
        nn = mat[a2[i]] + r[i]
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        b = nndist(ind[0])
        d = nndist(ind[1])
        h3 = triangle_height(nn[ind[0]], r[i])
        h4 = triangle_height(nn[ind[1]], r[i])

        nn = r[i] - nnmat[a1[i]]
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        a = nndist[ind[0]]
        e = nndist[ind[1]]
        h1 = triangle_height(nn[ind[0]], r[i])
        h2 = triangle_height(nn[ind[1]], r[i])

        l.append((a + b + d + e)/4)
        h.append((h1 + h2 + h3 + h4)/4)
    return {'t': list(tij), 'c': list(c), 'l': list(p), 'h': list(h)/4)}
