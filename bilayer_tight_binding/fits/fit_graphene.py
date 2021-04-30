import h5py
import numpy as np 
import pandas as pd
import statsmodels.api as sm
from bilayer_tight_binding.fits.training_data import graphene_training_data

# Single-layer graphene fits
df1, df2, df3 = graphene_training_data('../../datasets/graphene/')
df1_, df2_, df3_ = graphene_training_data('../../datasets/bilayer/')

fits = {
    't01': {
        'df': pd.concat([df1, df1_]),
    },
    't02': {
        'df': pd.concat([df2, df2_]),
    },
    't03': {
        'df': pd.concat([df3, df3_]),
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
