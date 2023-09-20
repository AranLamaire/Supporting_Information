#! /usr/bin/env python

import numpy as np
import h5py
from mpi4py import MPI
import sys, os

from molmod.units import *
from yaff import *
from yaff import log

from mylammps import *
#from tailcorr import ForcePartTailCorrection

from test_ghostatoms import *

from yaffdriver import YAFFDriver, MySocket

# Setup MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# Set random seed, important to get the same velocities for all processes
np.random.seed(5)

# Turn off logging for all processes, it can be turned on for one selected process later on
# log.set_level(log.silent)

def get_ff(rcut=12.0*angstrom, use_lammps=True, smooth_ei=False, supercell=(1,1,1), tailcorr=False):
    
    system, ghost_indexes = get_H2O_system('init.chk')
    ff = ForceField.generate(system, 'pars_long.txt', rcut=rcut, alpha_scale=3.2, gcut_scale=1.5, smooth_ei=smooth_ei)

    if tailcorr:
        for part in ff.parts:
            if part.name=='pair_mm3':
                sigmas = part.pair_pot.sigmas.copy()
                epsilons = part.pair_pot.epsilons.copy()
                onlypaulis = part.pair_pot.onlypaulis.copy()
                tr = part.pair_pot.get_truncation()
                pair_pot = PairPotMM3(sigmas, epsilons, onlypaulis, rcut, tr=tr)
                part = ForcePartPair(ff.system, ff.nlist, part.scalings, pair_pot)
                part_tail = ForcePartTailCorrection(ff.system, part.pair_pot)

            if part.name=='pair_lj':
                sigmas = part.pair_pot.sigmas.copy()
                epsilons = part.pair_pot.epsilons.copy()
                tr = part.pair_pot.get_truncation()
                pair_pot = PairPotLJ(sigmas, epsilons, rcut, tr=tr)
                part = ForcePartPair(ff.system, ff.nlist, part.scalings, pair_pot)
                part_tail = ForcePartTailCorrection(ff.system, part.pair_pot)
   
    if use_lammps:
        data_fn = 'lammps_%d_%d_%d.data'% (supercell[0], supercell[1], supercell[2])
        table_fn = 'rcut_%4.1f' % (rcut/angstrom)
        if smooth_ei: table_fn += '_smooth_ei'
        table_fn += '.table'
        if not os.path.isfile(data_fn):
            if rank==0:
                write_lammps_data(system, fn=data_fn)
        comm.Barrier()
        
        # Replace non-covalent part with LAMMPS
        for part in ff.parts:
            if part.name=='valence': part_valence = part
            elif part.name=='pair_ei': part_ei = part
        nlist = BondedNeighborList(system, N_host_atoms=len(system.numbers)-4*len(ghost_indexes))
        pair_pot = PairPotEI(system.charges, 0.0, part_ei.pair_pot.rcut, tr=None, radii=system.radii)
        scalings = Scalings(system, scale1=1.0, scale2=1.0, scale3=0.0)
        pair_gauss = ForcePartPair(system, nlist, scalings, pair_pot)
        pair_lammps = ForcePartLammps(system, pppm_accuracy=1e-5, scalings=[0.0,0.0,1.0,0.0,0.0,1.0], fn_system=data_fn, fn_table=table_fn, fn_log='/dev/null', comm=comm)
        ff = ForceField(system, [pair_lammps], nlist)
    
    if tailcorr:
        ff.add_part(part_tail)

    ff = GhostForceField(ff, ghost_indexes, write_ghost_positions_H2O, write_ghost_forces_H2O)

    # Remove intermolecular interactions water atoms
    for p, part in enumerate(ff.ff.parts):
        if part.name == 'pair_ei':
            #print ff.compute()
            stab = []
            for i, fftype in enumerate(ff.ff.system.ffatypes):
                if fftype == 'M':
                    index = i
            for j, fftype in enumerate(ff.ff.system.ffatype_ids):
                if fftype == index:

                    stab.append((j, j-3, 0.0, 1))
                    stab.append((j-1, j-3, 0.0, 1))
                    stab.append((j-2, j-3, 0.0, 1))
                    stab.append((j, j-2, 0.0, 2))
                    stab.append((j-1, j-2, 0.0, 2))
                    stab.append((j, j-1, 0.0, 2))
            stab.sort()
            stab = np.array(stab, dtype=scaling_dtype)
            #print stab
            ff.ff.parts[p].scalings.stab = stab
            #ff.ff.parts[p].scalings.check_mic(ff.ff.system)
            #print ff.compute()
        if part.name == 'ewald_cor':
            #print ff.compute()
            stab = []
            for i, fftype in enumerate(ff.ff.system.ffatypes):
                if fftype == 'M':
                    index = i
            for j, fftype in enumerate(ff.ff.system.ffatype_ids):
                if fftype == index:
                    stab.append((j, j-3, 0.0, 1))
                    stab.append((j-1, j-3, 0.0, 1))

                    stab.append((j-2, j-3, 0.0, 1))
                    stab.append((j, j-2, 0.0, 2))
                    stab.append((j-1, j-2, 0.0, 2))
                    stab.append((j, j-1, 0.0, 2))
            stab.sort()
            stab = np.array(stab, dtype=scaling_dtype)
            #print stab
            ff.ff.parts[p].scalings.stab = stab
            #ff.ff.parts[p].scalings.check_mic(ff.ff.system)
            #print ff.compute()
    
    return ff, system

if __name__=='__main__':

    rcut = 15.0*angstrom
    use_lammps = True
    smooth_ei = True
    supercell = (1,1,1)
    tailcorr = False

    ff, system = get_ff(rcut=rcut, use_lammps=use_lammps, smooth_ei=smooth_ei, supercell=supercell, tailcorr=tailcorr)

    servername = 'UiO66_160_H2O_Trotter_32_300K_long'

    logf = open('/dev/null', 'w')
    log._file = logf
    
    socket = MySocket(servername, verbose=False)
    driver = YAFFDriver(socket, ff)
    driver.run()
