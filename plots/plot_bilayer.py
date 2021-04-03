import sys
import h5py
import numpy as np 
import matplotlib.pyplot as plt
sys.path.append('../fits')
from training_data import bilayer_training_data
from functions import exponential, moon, fang

p = []
with h5py.File('../fits/fit_bilayer.hdf5','r') as hdf:
    p.append(list(hdf['exponential']['parameters']))
    p.append(list(hdf['moon']['parameters']))
    p.append(list(hdf['fang']['parameters']))
df = bilayer_training_data('../datasets/bilayer/')
data = {
    'exponential': {
        'function': exponential,
        'X': df['d'].values,
        'p': p[0],
    },
    'moon': {
        'function': moon,
        'X': [df['d'].values, df['dz'].values],
        'p': p[1],
    },
    'fang': {
        'function': fang,
        'X': [df['dxy'].values, df['theta_12'].values, df['theta_21'].values],
        'p': p[2],
    }
}

for k in data.keys():
    tpred = data[k]['function'](data[k]['X'], *data[k]['p'])
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (3,7))
    ax[0].plot(df['d'], df['t'], '.', label = 'DFT')
    ax[0].plot(df['d'], tpred, '.', label = 'Model')
    ax[1].plot(df['d'], df['t'] - tpred, '.')
    ax[1].axhline(0, c='k', ls='--')
    ax[1].set_xlabel('d (Bohr)')
    ax[1].set_ylabel(r'$t - t_{pred}$ (eV)')
    ax[1].set_ylim((-0.12, 0.12))
    ax[0].set_ylim((-0.1, 0.4))
    ax[0].set_ylabel('t (eV)')
    ax[0].legend(loc='best')
    plt.title(k)
    plt.savefig(k+'.pdf', bbox_inches='tight')
    plt.close()
