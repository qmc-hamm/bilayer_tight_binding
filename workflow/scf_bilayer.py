#!/usr/bin/env python
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf

if __name__ == '__main__':
    basis = 'pobtzvp'
    functional = 'GGA_XC_BEEFVDW'
    nk = 36

    a = [[ 4.649, 0.000, 0.000], 
         [-2.324, 4.026, 0.000], 
         [ 0.000, 0.000, 30.000]]
    atom =\
        '''
        C 0.0 0.000 11.752;
        C 0.0 2.684 11.752;
        C 0.0 2.684 18.249;
        C 0.0 5.368 18.249;
        '''

    cell = pbcgto.Cell()
    cell.build(
             unit = 'B',
             a = a,
             atom = atom,
             dimension = 2,
             verbose = 7,
             precision = 1e-6,
             basis = basis,
    )   
    
    calc_uuid = uuid.uuid1().hex
    mf = pbchf.KRKS(cell).mix_density_fit(auxbasis = 'weigend')
    mf.xc = functional
    mf.kpts = cell.make_kpts([nk, nk, 1]) 
    mf.level_shift = 0.1 
    mf.chkfile = 'bilayer.chk'
    mf.kernel()
