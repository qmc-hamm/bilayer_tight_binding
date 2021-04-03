import h5py
import numpy as np 
import statsmodels.api as sm
from scipy.optimize import curve_fit
from functions import exponential, moon, fang
from training_data import graphene_training_data, bilayer_training_data

# Single-layer graphene fits
df1, df2, df3 = graphene_training_data('../datasets/graphene/')
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
f = h5py.File('fit_graphene.hdf5','w')
for k in fits.keys():
    y = fits[k]['df']['t']
    X = sm.add_constant(fits[k]['df'].drop(['t'], axis = 1))
    lm = sm.OLS(y, X).fit()
    g = f.create_group(k)
    g.create_dataset('parameters', data=lm.params)
    g.create_dataset('standard_errors', data=lm.HC0_se)
f.close() 




# Bi-layer graphene fits
df = bilayer_training_data('../datasets/bilayer/')
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
f = h5py.File('fit_bilayer.hdf5','w')
for k in fits.keys():
    p, perr = curve_fit(fits[k]['function'], fits[k]['X'], df['t'], p0=fits[k]['p0'])
    g = f.create_group(k)
    g.create_dataset('parameters', data=p)
    g.create_dataset('standard_errors', data=np.sqrt(np.diag(perr)))
f.close()