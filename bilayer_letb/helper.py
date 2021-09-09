import numpy as np
import ase
import pythtb
import bilayer_letb.model
from scipy.spatial.distance import cdist

def compute_hoppings(lattice_vectors, atomic_basis, hopping_model):
    """
    Compute hoppings in a hexagonal environment of the computation cell 
    Adequate for large unit cells (> 100 atoms)
    Input:
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for graphene in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        hopping_model   - model for computing hoppings

    Output:
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """

    natom = len(atomic_basis)
    di = []
    dj = []
    extended_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, np.newaxis] * dx + lattice_vectors[1, np.newaxis] * dy)
            di += [dx] * natom
            dj += [dy] * natom
    distances = cdist(atomic_basis, extended_coords)
    indi, indj = np.where((distances > 0) & (distances < 10)) # 10 Bohr cutoff
    di = np.array(di)[indj]
    dj = np.array(dj)[indj]
    i  = indi
    j  = indj % natom
    hoppings = hopping_model(lattice_vectors, atomic_basis, i, j, di, dj) / 2 # Divide by 2 since we are double counting every pair
    return i, j, di, dj, hoppings

def pythtb_model(ase_atoms:ase.Atoms, model_type='letb'):
    """
    Returns a pythtb model object for a given ASE atomic configuration 
    Input:
        ase_atoms - ASE object for the periodic system
        model_type - 'letb' or 'mk'
    Output:
        gra - PythTB model describing hoppings between atoms using model_type        
    """
    if model_type not in ['letb','mk']:
        print("Invalid function {}".format(model_functions))
        return None

    models_functions = {'letb':bilayer_letb.model.letb,
                         'mk':bilayer_letb.model.mk}

    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = np.asarray(ase_atoms.cell)*conversion
    atomic_basis = np.asarray(ase_atoms.get_positions(wrap=True))*conversion

    i, j, di, dj, hoppings = compute_hoppings(lattice_vectors, atomic_basis, models_functions[model_type])
    gra = pythtb.tb_model(2, 3, lattice_vectors, atomic_basis)
    for ii, jj, dii, djj, hopping in zip(i, j, di, dj, hoppings):
        gra._hoppings.append([hopping, ii, jj, np.array([dii, djj, 0])])
    return gra
