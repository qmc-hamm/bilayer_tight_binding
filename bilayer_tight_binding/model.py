import sys
import h5py
import numpy as np 
from bilayer_tight_binding.fits.functions import fang
from bilayer_tight_binding.descriptors import descriptors_graphene, descriptors_bilayer

def load_graphene_fit():
    """
    Load in the graphene fit from ../fits/fit_graphene.hdf5
    """
    fit = {}
    with h5py.File('../../fits/fit_graphene.hdf5','r') as hdf:
        fit['t01'] = list(hdf['t01']['parameters'])
        fit['t02'] = list(hdf['t02']['parameters'])
        fit['t03'] = list(hdf['t03']['parameters'])
    return fit

def load_bilayer_fit():
    fit = {}
    with h5py.File('../../fits/fit_bilayer.hdf5','r') as hdf:
        fit['fang'] = list(hdf['fang']['parameters'])
    return fit

def graphene(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for graphene in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj in eV
    """
    # Extend lattice_vectors to (3 x 3) for our descriptors, the third lattice vector is arbitrary
    latt_vecs = np.append(lattice_vectors, [[0, 0, 0]], axis = 0)
    atomic_basis = np.array(atomic_basis)
    i = np.array(i)
    j = np.array(j)
    di = np.array(di)
    dj = np.array(dj)

    # Get the descriptors for the fit models
    partition   = descriptors_graphene.partition_tb(lattice_vectors, atomic_basis, di, dj, i, j)
    descriptors = descriptors_graphene.descriptors(lattice_vectors, atomic_basis, di, dj, i, j)

    # Get the fit model parameters
    fit = load_graphene_fit()

    # Predict hoppings
    t01 = np.dot(descriptors[0], fit['t01'][1:]) + fit['t01'][0]
    t02 = np.dot(descriptors[1], fit['t02'][1:]) + fit['t02'][0]
    t03 = np.dot(descriptors[2], fit['t03'][1:]) + fit['t03'][0]

    # Reorganize
    hoppings = np.zeros(len(i))
    hoppings[partition[0]] = t01
    hoppings[partition[1]] = t02
    hoppings[partition[2]] = t03

    return hoppings

def bilayer(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for graphene in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    # Extend lattice_vectors to (3 x 3) for our descriptors, the third lattice vector is arbitrary
    latt_vecs = np.append(lattice_vectors, [[0, 0, 0]], axis = 0)
    atomic_basis = np.array(atomic_basis)
    i = np.array(i)
    j = np.array(j)
    di = np.array(di)
    dj = np.array(dj)

    # Get the bi-layer descriptors 
    descriptors = descriptors_bilayer.descriptors(lattice_vectors, atomic_basis, di, dj, i, j)
    
    # Partition the intra- and inter-layer hoppings indices 
    interlayer = np.array(descriptors['dz'] > 0)
    
    # Compute the inter-layer hoppings
    fit = load_bilayer_fit()
    X = descriptors[['dxy','theta_12','theta_21']].values[interlayer]
    interlayer_hoppings = fang(X.T, *fit['fang'])

    # Compute the intra-layer hoppings
    intralayer_hoppings = graphene(lattice_vectors, atomic_basis, i[~interlayer], j[~interlayer], di[~interlayer], dj[~interlayer])

    # Reorganize
    hoppings = np.zeros(len(i))
    hoppings[interlayer] = interlayer_hoppings
    hoppings[~interlayer] = intralayer_hoppings

    return hoppings
