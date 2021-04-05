import h5py
import numpy as np 
import statsmodels.api as sm
from bilayer_tight_binding.fits.training_data import graphene_training_data

# Single-layer graphene fits
df1, df2, df3 = graphene_training_data('../../datasets/graphene/')
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
