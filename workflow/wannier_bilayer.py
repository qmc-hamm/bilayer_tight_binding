#!/usr/bin/env python
import numpy as np 
from pyscf.pbc import scf as pbchf
from pyscf.pbc import lib as pbclib
from pyscf.pbc.tools import pywannier90

chk = 'bilayer.chk'
cell = pbclib.chkfile.load_cell(chk)
mfdict = pbclib.chkfile.load(chk, 'scf')

nk = 36
mf = pbchf.KROKS(cell).mix_density_fit(auxbasis = 'weigend')
mf.xc = 'PBE'
mf.kpts = cell.make_kpts([nk, nk, 1]) 
mf.level_shift = 0.1 
mf.mo_energy = np.array(mfdict['mo_energy'])
mf.mo_coeff  = np.array(mfdict['mo_coeff'])

num_wann = 4
keywords = ''' 
exclude_bands:  1 - 4, 20 - 72 
begin projections
C:pz
end projections 
dis_win_min =  -12.00
dis_win_max =    8.00
write_xyz = True
write_hr = True
'''
w90 = pywannier90.W90(mf, cell, [nk, nk, 1], num_wann, other_keywords = keywords)
w90.make_win()
w90.setup()
w90.export_unk(grid = [25, 25, 25])
w90.kernel()
