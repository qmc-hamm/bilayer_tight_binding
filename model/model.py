import sys
import numpy as np 
import pandas as pd
sys.path.apend('../fits')
sys.path.append('../descriptors')
from functions import fang
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
    fit = {}
    with h5py.File('../fits/fit_bilayer.hdf5','w') as hdf:
        fit['fang'] = list(hdf['fang']['parameters'])
    return fit

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
    t01 = np.dot(fit['t01'][1:], descriptors) + fit['t01'][0]
    t02 = np.dot(fit['t02'][1:], descriptors) + fit['t02'][0]
    t03 = np.dot(fit['t03'][1:], descriptors) + fit['t03'][0]

    # Reorganize
    hoppings = np.zeros(len(i))
    hoppings[partition[0]] = t01
    hoppings[partition[1]] = t02
    hoppints[partition[2]] = t03

    return hoppings

def bilayer(lattice_vectors, atomic_basis, i, j, di, dj):
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

    # Get the bi-layer descriptors 
    descriptors = descriptors_bilayer.descriptors(lattice_vectors, atomic_basis, di, dj, i, j)
    
    # Partition the intra- and inter-layer hoppings indices 
    interlayer = np.array(descriptors['dz'] > 0)
    
    # Compute the inter-layer hoppings
    fit = load_bilayer_fit()
    X = descriptors[['dxy','theta_12','theta_21']].values[interlayer]
    interlayer_hoppings = fang(X, *fit['fang'])

    # Compute the intra-layer hoppings
    intralayer_hoppings = graphene(lattice_vectors, atomic_basis, i[~interlayer], j[~interlayer], di[~interlayer], dj[~interlayer])

    # Reorganize
    hoppings = np.zeros(len(i))
    hoppings[interlayer] = interlayer_hoppings
    hoppings[~interlayer] = intralayer_hoppings

    return hoppings
