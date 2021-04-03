import sys
import h5py
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt
sys.path.append('../fits')
from training_data import graphene_training_data

p = []
with h5py.File('../fits/fit_graphene.hdf5','r') as hdf:
    p.append(list(hdf['t01']['parameters']))
    p.append(list(hdf['t02']['parameters']))
    p.append(list(hdf['t03']['parameters']))
df = graphene_training_data('../datasets/graphene/')
data = {
    't01': {
        'df': df[0],
        'p':  p[0],
    },
    't02': {
        'df': df[1],
        'p':  p[1],
    },
    't03': {
        'df': df[2],
        'p':  p[2],
    }
}

for k in data.keys():
    t = data[k]['df']['t']
    X = sm.add_constant(data[k]['df'].drop(['t'], axis = 1)).values
    tpred = np.dot(X, data[k]['p'])
    
    fig, ax = plt.subplots(figsize = (3,3))
    ax.plot(t, tpred, '.')
    ax.plot(t, t, 'k--')
    ax.set_xlabel('t (eV)')
    ax.set_ylabel(r'$t_{pred}$ (eV)')
    ax.set_title(k)
    plt.savefig(k+'.pdf', bbox_inches='tight')
    plt.close()
