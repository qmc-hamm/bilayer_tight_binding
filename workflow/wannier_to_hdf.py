import h5py
import numpy as np 
from pythtb import w90

def wannier_to_hdf(fpath, outfile):
    """
    Given the fpath for the wannier output files,
    collect data from the wannier files to outfile.hdf
    """

    model_container = w90(fpath, 'wannier90')
    model = model_container.model(max_distance = 6) # 6 Angstrom
    
    tb_hamiltonian = {
        'tij': [],
        'displacementi': [],
        'displacementj': [],
        'atomi': [], 
        'atomj': [],
        'energy': [],
    }
    for hopping in model._hoppings:
        tb_hamiltonian['tij'].append(np.real(hopping[0]))
        tb_hamiltonian['displacementi'].append(hopping[3][0])
        tb_hamiltonian['displacementj'].append(hopping[3][1])
        tb_hamiltonian['atomi'].append(hopping[1])
        tb_hamiltonian['atomj'].append(hopping[2])

    # Make HDF5
    f = h5py.File(outfile+'.hdf5','w')
    f.create_dataset('uuid', data=uuid.encode('utf-8'))
    f.create_dataset('energy', data=d.iloc[calc]['energy'])
    f.create_dataset('lattice_vectors', data=model._lat)
    f.create_dataset('atomic_basis', data=np.dot(model._orb, model._lat))
    tb_group = f.create_group("tb_hamiltonian")
    for tb_key in tb_hamiltonian.keys():
        tb_group.create_dataset(tb_key, data=tb_hamiltonian[tb_key])
    f.close() 
