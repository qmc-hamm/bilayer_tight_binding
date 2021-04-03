import h5py
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from descriptors_bilayer import descriptors 
from functions_bilayer import exponential, moon, fang 

# Construct pandas data frame for fitting
data = []
flist = subprocess.Popen(["ls", '../datasets/bilayer/'],
                      stdout=subprocess.PIPE).communicate()[0]
flist = flist.decode('utf-8').split("\n")[:-1]
flist = ['../datasets/bilayer/'+x for x in flist]
for f in flist:
    data.append(descriptors(f))
df = pd.concat(data)
df = df[df['dz'] > 0] # Inter-hoppings only

# Conduct fits
fits = {
    'exponential': {
        'function': exponential,
        'X':  df['d'].values,
        'p0': [0.300, 2.3], 
    },
    'moon': {
        'function': moon,
        'X':  [df['d'].values, df['dz'].values],
        'p0': [-100, 0.300, 1.0],
    },
    'fang': {
        'function': fang,
        'X':  [df['dxy'].values, df['theta_12'].values, df['theta_21'].values],
        'p0': [0.3155, 1.7543, 2.0010, -0.0688, 3.4692, 0.5212, -0.0083, 2.8764, 1.5206, 1.5731],
    },
}
for k in fits.keys():
    p, perr = curve_fit(fits[k]['function'], fits[k]['X'], df['t'], p0=fits[k]['p0'])
    fits[k]['parameters'] = p
    fits[k]['standard_errors'] = np.sqrt(np.diag(perr))

# Save to HDF output file 
f = h5py.File('fit_bilayer.hdf5','w')
for k in fits.keys():
    g = f.create_group(k)
    g.create_dataset('parameters', data=fits[k]['parameters'])
    g.create_dataset('standard_errors', data=fits[k]['standard_errors'])
f.close() 
