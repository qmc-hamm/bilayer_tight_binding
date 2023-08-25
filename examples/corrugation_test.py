# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:13:20 2023

@author: danpa
"""

import sys
import json
import ase
from pythtb import *
import matplotlib.pyplot as plt
import numpy as np
from bilayer_letb.api import tb_model

def plot_bands(atoms,colors=['black'],labels=[""],title="",figname='test'):
    #create model
    if type(atoms) != list:
        atoms = [atoms]
    fig, ax = plt.subplots(figsize = (3,3))
    for j,a in enumerate(atoms):
        letb = tb_model(a,model_type='letb')
        
        # Compute bands
        k=[[1/3., 2/3.], [0.0, 0.0], [0.5, 0.0], [2/3., 1/3.]]
        (k_vec, k_dist, k_node) = letb.k_path(k, 100)
        evals = letb.solve_all(k_vec)
        evals -= min(evals[int(len(atomic_basis)/2), :])
        
        # plot band structure
        
        for i in range(a.get_global_number_of_atoms()):
            if i==0:
                ax.plot(k_dist, evals[i,:], color=colors[j],label=labels[j])
            else:
                 ax.plot(k_dist, evals[i,:], color=colors[j])
        
    # figure formatting
    ax.set_ylabel(r'$E - E_F$ (eV)')
    if labels[0]!="":
        ax.legend()
    ax.set_title(title)
    ax.set_ylim((-2, 2))
    ax.set_xticks(k_node)
    ax.set_xticklabels(["K", "$\Gamma$", "M", "$K^\prime$"])
    ax.set_xlim(k_node[0], k_node[-1])
    fig.savefig(figname, bbox_inches='tight')

if __name__=="__main__":
    import ase.io
    #no corrugation, original LETB code
    dump_atoms = ase.io.read('dump.theta_9_43',format='lammps-dump-text',index=":")
    ase_atoms_nopatch= dump_atoms[0]
    
    lattice_vectors = ase_atoms_nopatch.get_cell()
    atomic_basis = ase_atoms_nopatch.positions
    # red_coord = ase_atoms.get_scaled_positions()
    d = json.load(open('tblg.json'))
    lattice_vectors = d['latvec']
    ase_atoms_nopatch.set_cell(lattice_vectors)
    # atomic_basis = np.array(d['atoms'])
    # ase_atoms = ase.Atoms(['C']*len(atomic_basis), 
    #                   positions = atomic_basis, cell = lattice_vectors, pbc = True)
    
    # print(ase_atoms.has('layer_types'))
    
    
    plot_bands(ase_atoms_nopatch,title=r"$\theta$=9.42, no corrugation, no letb patch",
                figname='no_corrugation_no_patch.png')
    
    
    # #no corrugation, patched LETB code
    # ##set layer types to differentiate between interlayer and intralayer interactions
    ase_atoms = ase_atoms_nopatch.copy()
    z = np.array(atomic_basis)[:,2]
    top_layer_ind = np.squeeze(np.where(z>np.mean(z)))
    bot_layer_ind = np.squeeze(np.where(z<np.mean(z)))
    layer_types = np.zeros(np.shape(atomic_basis)[0],dtype=np.int64)
    layer_types[top_layer_ind] = np.zeros_like(top_layer_ind,dtype=np.int64)
    layer_types[bot_layer_ind] = np.ones_like(bot_layer_ind,dtype=np.int64) 
    sym = ['B' for i in range(ase_atoms.get_global_number_of_atoms())]
    for ind in top_layer_ind: sym[ind]='Ti'
    ase_atoms.set_array('layer_types',layer_types)
    ase_atoms.set_array('mol-id',layer_types)
    ase_atoms.set_chemical_symbols(sym)
    ase.io.write('theta_9_43.data',ase_atoms,format='lammps-data',atom_style='full')
    plot_bands(ase_atoms,title=r"$\theta$=9.42, no corrugation, with letb patch",
                figname='no_corrugation_with_patch.png')
    
    #high corrugation, original LETB code
    amplitude = 20 #angstroms
    final_atom_pos = np.array(dump_atoms[-1].positions)
    atomic_basis[top_layer_ind,2] += amplitude*(final_atom_pos[top_layer_ind,2]-atomic_basis[top_layer_ind,2]) #*np.sum(np.cos(2*np.pi*red_coord[top_layer_ind,:2]),axis=1) + np.mean(atomic_basis[top_layer_ind,2])
    atomic_basis[bot_layer_ind,2] -= amplitude*(final_atom_pos[bot_layer_ind,2]-atomic_basis[bot_layer_ind,2]) #*np.sum(np.cos(2*np.pi*red_coord[bot_layer_ind,:2]),axis=1) + np.mean(atomic_basis[bot_layer_ind,2])
    
    plt.clf()
    plt.scatter(atomic_basis[top_layer_ind,0],atomic_basis[top_layer_ind,1],c=atomic_basis[top_layer_ind,2])
    plt.colorbar()
    plt.show()
    plt.savefig("high_corr_struct.png")
    plt.clf()
    ase_atoms_nopatch = ase.Atoms(['C']*len(atomic_basis), 
                          positions = atomic_basis, cell = lattice_vectors, pbc = True)
    
    ase_atoms= ase_atoms_nopatch.copy()
    ase_atoms.set_array('layer_types',layer_types)
    colors = ['red','black']
    labels = ['no patch','with patch']
    plot_bands([ase_atoms_nopatch,ase_atoms],colors=colors,labels=labels,title=r"$\theta$=9.42, corrugation = 2.5 Angstroms, 20x",
                figname='20x_corrugation_compare.png')
    
    # #high corrugation, patched LETB code
    
    
    # plot_bands(ase_atoms,title=r"$\theta$=9.42, corrugation = 10 Angstroms, with letb patch",
    #             figname='10_corrugation_with_patch.png')
    
