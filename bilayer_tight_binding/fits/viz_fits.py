import h5py
import matplotlib.pyplot as plt 
import numpy as np

with h5py.File('fit_graphene.hdf5','r') as hdf:
    for fit in ['t01','t02','t03']:
        ptest = list(hdf[fit]['parameters_test'])
        ptrain = list(hdf[fit]['parameters_train'])

        for pt in ptest:
            plt.plot(pt, 'ko-')
        for ptr in ptrain:
            plt.plot(ptr, 'bo-')
        plt.title(fit + ' black = Test, blue = Train')
        plt.xlabel('Parameter')
        plt.ylabel('Value (eV)')
        plt.show()

with h5py.File('fit_bilayer.hdf5','r') as hdf:
    for fit in ['exponential','moon','fang']:
        ptest = list(hdf[fit]['parameters_test'])
        ptrain = list(hdf[fit]['parameters_train'])

        for pt in ptest:
            plt.plot(pt, 'ko-')
        for ptr in ptrain:
            plt.plot(ptr, 'bo-')
        plt.title(fit + ' black = Test, blue = Train')
        plt.xlabel('Parameter')
        plt.ylabel('Value (eV)')
        plt.show()
