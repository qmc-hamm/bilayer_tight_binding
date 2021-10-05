import h5py
import subprocess
import numpy as np
import pandas as pd
from bilayer_letb.descriptors import descriptors_interlayer, descriptors_intralayer 

def intralayer_training_data(dataset):
    data = []
    flist = subprocess.Popen(["ls", dataset],
                          stdout=subprocess.PIPE).communicate()[0]
    flist = flist.decode('utf-8').split("\n")[:-1]
    flist = [dataset+x for x in flist]

    df1 = []
    df2 = []
    df3 = []
    for f in flist:
        with h5py.File(f, 'r') as hdf:
            # Unpack hdf
            lattice_vectors = np.array(hdf['lattice_vectors'][:]) * 1.88973
            atomic_basis =    np.array(hdf['atomic_basis'][:])    * 1.88973
            tb_hamiltonian = hdf['tb_hamiltonian']
            tij = np.array(tb_hamiltonian['tij'][:])
            di  = np.array(tb_hamiltonian['displacementi'][:])
            dj  = np.array(tb_hamiltonian['displacementj'][:])
            ai  = np.array(tb_hamiltonian['atomi'][:])
            aj  = np.array(tb_hamiltonian['atomj'][:])
        partition = descriptors_intralayer.partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj) 
        data = descriptors_intralayer.descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj) 
        data[0]['t'] = tij[partition[0]]
        data[1]['t'] = tij[partition[1]]
        data[2]['t'] = tij[partition[2]]
        df1.append(data[0])
        df2.append(data[1])
        df3.append(data[2])
    df1 = pd.concat(df1)
    df2 = pd.concat(df2)
    df3 = pd.concat(df3)
    return df1, df2, df3

def interlayer_training_data(dataset):
    data = []
    flist = subprocess.Popen(["ls", dataset],
                          stdout=subprocess.PIPE).communicate()[0]
    flist = flist.decode('utf-8').split("\n")[:-1]
    flist = [dataset+x for x in flist]

    df = []
    for f in flist:
        if ".hdf5" in f:
            with h5py.File(f, 'r') as hdf:
                # Unpack hdf
                lattice_vectors = np.array(hdf['lattice_vectors'][:]) * 1.88973
                atomic_basis =    np.array(hdf['atomic_basis'][:])    * 1.88973
                tb_hamiltonian = hdf['tb_hamiltonian']
                tij = np.array(tb_hamiltonian['tij'][:])
                di  = np.array(tb_hamiltonian['displacementi'][:])
                dj  = np.array(tb_hamiltonian['displacementj'][:])
                ai  = np.array(tb_hamiltonian['atomi'][:])
                aj  = np.array(tb_hamiltonian['atomj'][:])
            data = descriptors_interlayer.descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj) 
            data['t'] = tij 
            df.append(data)
    df = pd.concat(df)
    df = df[df['dz'] > 1] # Inter-layer hoppings only, allows for buckling
    return df
