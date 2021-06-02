import sys
import h5py
import numpy as np 
import bilayer_tight_binding.fits
from bilayer_tight_binding.fits.functions import fang, moon
from bilayer_tight_binding.descriptors import descriptors_graphene, descriptors_bilayer

def load_graphene_fit():
    # Load in fits, average over k-folds
    fit = {}
    f = "/".join(bilayer_tight_binding.fits.__file__.split("/")[:-1])+"/fit_graphene.hdf5"
    with h5py.File(f,'r') as hdf:
        fit['t01'] = np.array(list(hdf['t01']['parameters_test'])).mean(axis = 0)
        fit['t02'] = np.array(list(hdf['t02']['parameters_test'])).mean(axis = 0)
        fit['t03'] = np.array(list(hdf['t03']['parameters_test'])).mean(axis = 0)
    return fit

def load_bilayer_fit():
    # Load in fits, average over k-folds
    fit = {}
    f = "/".join(bilayer_tight_binding.fits.__file__.split("/")[:-1])+"/fit_bilayer.hdf5"
    with h5py.File(f,'r') as hdf:
        fit['fang'] = np.array(list(hdf['fang']['parameters_test'])).mean(axis = 0)
    return fit

def graphene(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    Our model for single layer graphene
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
    hoppings[partition[3]] = 0

    return hoppings

def bilayer(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    Our model for bilayer graphene
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
    interlayer = np.array(descriptors['dz'] > 1) # Allows for buckling, won't be more than 1 Bohr
    
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

def moon_bilayer(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    Moon model for bilayer graphene - Moon and Koshino, PRB 85 (2012)
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for graphene in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    lattice_vectors = np.array(lattice_vectors)
    atomic_basis = np.array(atomic_basis)
    i = np.array(i)
    j = np.array(j)
    di = np.array(di)
    dj = np.array(dj)
    dxy, dz = descriptors_graphene.ix_to_dist(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = moon([np.sqrt(dz**2 + dxy**2), dz], -2.7, 1.17, 0.48)
    return hoppings
