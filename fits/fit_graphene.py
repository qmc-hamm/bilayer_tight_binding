import h5py
import subprocess
import numpy as np
import pandas as pd
import statsmodels.api as sm
from descriptors_graphene import partition_tb, descriptors 

# Construct pandas data frame for fitting
data = []
flist = subprocess.Popen(["ls", '../datasets/graphene/'],
                      stdout=subprocess.PIPE).communicate()[0]
flist = flist.decode('utf-8').split("\n")[:-1]
flist = ['../datasets/graphene/'+x for x in flist]

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
    partition = partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj)
    data = descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj)
    data[0]['t'] = tij[partition[0]]
    data[1]['t'] = tij[partition[1]]
    data[2]['t'] = tij[partition[2]]
    df1.append(data[0])
    df2.append(data[1])
    df3.append(data[2])
df1 = pd.concat(df1)
df2 = pd.concat(df2)
df3 = pd.concat(df3)

# Conduct fits
fits = {
    't01': {
        'df': df1,
    },
    't02': {
        'df': df2,
    },
    't03': {
        'df': df3,
    },
}
for k in fits.keys():
    y = fits[k]['df']['t']
    X = sm.add_constant(fits[k]['df'].drop(['t'], axis = 1))
    lm = sm.OLS(y, X).fit()
    fits[k]['parameters'] = lm.params
    fits[k]['standard_errors'] = lm.HC0_se
    print(lm.summary())

# Save to HDF output file 
f = h5py.File('fit_graphene.hdf5','w')
for k in fits.keys():
    g = f.create_group(k)
    g.create_dataset('parameters', data=fits[k]['parameters'])
    g.create_dataset('standard_errors', data=fits[k]['standard_errors'])
f.close() 
