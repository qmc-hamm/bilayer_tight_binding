import h5py
import numpy as np 
from scipy.optimize import curve_fit
from bilayer_tight_binding.fits.functions import exponential, moon, fang
from bilayer_tight_binding.fits.training_data import bilayer_training_data
from sklearn.model_selection import KFold

df = bilayer_training_data('../../datasets/bilayer/')
fits = { 
    'exponential': {
        'function': exponential,
        'X':  df['d'].values,
        'p0': [0.300, 2.3], 
    },  
    'moon': {
        'function': moon,
        'X':  np.array([df['d'].values, df['dz'].values]).T,
        'p0': [-100, 0.300, 1.0],
    },  
    'fang': {
        'function': fang,
        'X':  np.array([df['dxy'].values, df['theta_12'].values, df['theta_21'].values]).T,
        'p0': [0.3155, 1.7543, 2.0010, -0.0688, 3.4692, 0.5212, -0.0083, 2.8764, 1.5206, 1.5731],
    },
}

f = h5py.File('fit_bilayer.hdf5','w')
for k in fits.keys():
    ptest_list = []
    ptrain_list = []

    # 3-fold cross validation 
    kf = KFold(n_splits = 3)
    for train_index, test_index in kf.split(df):
        X_train, X_test = fits[k]['X'][train_index], fits[k]['X'][test_index]
        y_train, y_test = df['t'].values[train_index], df['t'].values[test_index]
        ptrain, _ = curve_fit(fits[k]['function'], X_train.T, y_train, p0=fits[k]['p0'])
        ptest,  _ = curve_fit(fits[k]['function'], X_test.T,  y_test,  p0=fits[k]['p0'])
        ptrain_list.append(ptrain)
        ptest_list.append(ptest)

    g = f.create_group(k)
    g.create_dataset('parameters_train', data=ptrain_list)
    g.create_dataset('parameters_test',  data=ptest_list)
f.close()
