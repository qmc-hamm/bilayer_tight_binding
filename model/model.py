import sys
import numpy as np 
import pandas as pd
sys.path.append('../descriptors')

import descriptors_graphene, descriptors_bilayer

def load_graphene_fit():
    """
    Load in the graphene fit from ../fits/fit_graphene.hdf5
    """
    fit = {}
    with h5py.File('../fits/fit_graphene.hdf5','w') as hdf:
        fit['t01'] = list(hdf['t01']['parameters'])
        fit['t02'] = list(hdf['t02']['parameters'])
        fit['t03'] = list(hdf['t03']['parameters'])
    return fit

def load_bilayer_fit():
    return

def graphene(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for graphene
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell 
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    # Extend lattice_vectors to (3 x 3) for our descriptors, the third lattice vector is arbitrary
    latt_vecs = np.append(lattice_vectors, [[0, 0, 25]], axis = 0)

    # Get the descriptors for the fit models
    partition   = descriptors_graphene.partition(lattice_vectors, di, dj, i, j)
    descriptors = descriptors_graphene.descriptors(lattice_vectors, atomic_basis, di, dj, i, j)

    # Get the fit model parameters
    fit = load_graphene_fit()

    # Predict hoppings
    t01 = np.dot(fit[0][1:], descriptors) + fit[0][0]
    t02 = np.dot(fit[1][1:], descriptors) + fit[1][0]
    t03 = np.dot(fit[2][1:], descriptors) + fit[2][0]

    # Reorganize
    hoppings = np.zeros(len(i))
    hoppings[partition[0]] = t01
    hoppings[partition[1]] = t02
    hoppints[partition[2]] = t03

    return hoppings

