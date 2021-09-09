import h5py
import numpy as np 
import pandas as pd
import statsmodels.api as sm
from training_data import graphene_training_data
from sklearn.model_selection import KFold

# Single-layer graphene fits
df1, df2, df3 = graphene_training_data('../data/')

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
f = h5py.File('../bilayer_tight_binding/parameters/fit_graphene.hdf5','w')
for k in fits.keys():
    y = fits[k]['df']['t']
    X = sm.add_constant(fits[k]['df'].drop(['t'], axis = 1))

    ptest_list = []
    ptrain_list = []

    kf = KFold(n_splits = 4, shuffle = True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lmtrain = sm.OLS(y_train, X_train).fit()
        lmtest = sm.OLS(y_test, X_test).fit()
        ptest_list.append(lmtest.params)
        ptrain_list.append(lmtrain.params)

    g = f.create_group(k)
    g.create_dataset('parameters_train', data=ptrain_list)
    g.create_dataset('parameters_test',  data=ptest_list)
f.close()
