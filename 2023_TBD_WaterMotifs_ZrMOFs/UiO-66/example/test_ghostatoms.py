#!/usr/bin/env python
# Original script: Steven Vandenbrande
# Adaptations to q-TIP4P/F: Aran Lamaire

import numpy as np
np.random.seed(3)

from yaff import System, ForceField, NeighborList, Scalings, PairPotLJ,\
 ForcePartPair, ForcePart, log
from molmod.io import load_chk
from molmod.units import angstrom, kcalmol, electronvolt

class GhostForceField(ForceField):
    '''A force field that implicitly contains ghost atoms. Ghost atoms do not
    play a role in sampling (Verlet, optimization, ...) but influence the energy.
    The position of ghost atoms is based on geometric rules.'''
    def __init__(self, ff, ghost_indexes, write_ghost_pos, write_ghost_gpos):
        """

           **Arguments**

           ff
                An instance of the ``ForceField`` class, containing all atoms
                (including ghost atoms). This is used for the energy evaluation

           ghost_indexes
                NumPy array containing the indexes of the ghost atoms

           write_ghost_pos
                A function that updates the positions of the ghost atoms based
                on the positions of the other atoms. The arguments of this
                function are a NumPy array with positions of ALL atoms and the
                indexes of the ghost atoms. This function writes directly to
                the NumPy array with positions.

        """
        self.ghost_indexes = ghost_indexes
        self.atom_indexes = np.array([iatom for iatom in xrange(ff.system.natom) if not iatom in self.ghost_indexes], dtype=int)
        self.system = ff.system.subsystem(self.atom_indexes)
        self.write_ghost_pos = write_ghost_pos
        self.write_ghost_gpos = write_ghost_gpos
        ForcePart.__init__(self, 'ghostff', self.system)
        self.ff = ff
        self.gpos_full = np.zeros((self.ff.system.pos.shape[0], self.ff.system.pos.shape[1]))
        self.parts = [self.ff]
        # These attributes need to be defined but are never used...
        self.gpos_parts = None
        self.vtens_parts = None
        if log.do_medium:
            with log.section('FFINIT'):
                log('Force field with %d ghost atoms and %d real atoms.' % (self.ghost_indexes.shape[0], self.atom_indexes.shape[0]))

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        ForcePart.update_rvecs(self.ff, rvecs)
        self.system.cell.update_rvecs(rvecs)
        self.ff.system.cell.update_rvecs(rvecs)
        if self.ff.nlist is not None:
            self.ff.nlist.update_rmax()
            self.ff.needs_nlist_update = True

    def update_pos(self, pos):
        '''See :meth:`yaff.pes.ff.ForcePart.update_pos`'''
        ForcePart.update_pos(self, pos)
        ForcePart.update_pos(self.ff, pos)
        self.system.pos[:] = pos
        self.ff.system.pos[self.atom_indexes] = pos
        # Call function that computes the positions of the ghost atoms based
        # on position of other atoms
        self.write_ghost_pos(self.ff.system.pos, self.ghost_indexes, self.system.cell)
        if self.ff.nlist is not None:
            self.ff.needs_nlist_update = True

    def _internal_compute(self, gpos, vtens):
        if gpos is None:
            my_gpos = None
        else:
            my_gpos = self.gpos_full
            my_gpos[:] = 0.0
        result = self.ff.compute(my_gpos, vtens)
        if gpos is not None:
            if np.isnan(my_gpos).any():
                raise ValueError('Some gpos element(s) is/are not-a-number (nan).')
            self.write_ghost_gpos(my_gpos, self.ghost_indexes)
            gpos += my_gpos[self.atom_indexes]
        return result

def get_H2O_system(file_name):
    gamma = 0.73612 # Gamma parameter of the q-TIP4P/F force field
    # The ghost atom gets atomic number 99, please do not use Einsteinium(99)
    # in these simulations :)
    chk = load_chk(file_name)

    if chk['ffatypes'].shape == chk['numbers'].shape:

        ghost_indexes = np.argwhere(chk['ffatypes']=='M').ravel()
        # Construct the Yaff system including ghost atoms
        system = System(chk['numbers'], chk['pos'], rvecs=chk['rvecs'], bonds=chk['bonds'], ffatypes=chk['ffatypes'])

    else:

        ffatype_id_ghost_atom = np.where(chk['ffatypes']=='M')[0]
        ghost_indexes = np.argwhere(chk['ffatype_ids']==ffatype_id_ghost_atom).ravel()
        # Construct the Yaff system including ghost atoms
        system = System(chk['numbers'], chk['pos'], rvecs=chk['rvecs'], bonds=chk['bonds'], ffatypes=chk['ffatypes'], ffatype_ids=chk['ffatype_ids'])

    return system, ghost_indexes

def write_ghost_positions_H2O(pos, ghost_indexes, cell):
    # In q-TIP4P/F, the position of each ghost atom depends on the coordinates of the three atoms
    # that precede the ghost atom in the system, i.e. the three atoms of the H2O molecule.
    gamma = 0.73612 # Gamma parameter of the q-TIP4P/F force field
    for iatom in ghost_indexes:
        # Find vectors connecting O-H atoms, taking PBC into account
        # TODO: skip mic for non-periodic systems
        r10 = pos[iatom-2]-pos[iatom-3]
        cell.mic(r10)
        r20 = pos[iatom-1]-pos[iatom-3]
        cell.mic(r20)
        pos[iatom] = pos[iatom-3] + 0.5*(1.0-gamma)*(r10+r20)

def write_ghost_forces_H2O(gpos, ghost_indexes):
    # In q-TIP4P/F, the position of each ghost atom depends on the coordinates of the three atoms
    # that precede the ghost atom in the system, i.e. the three atoms of the H2O molecule.
    # Application of the chain rule then yields an additional contribution to the gradient.
    gamma = 0.73612 # Gamma parameter of the q-TIP4P/F force field
    for iatom in ghost_indexes:
        gpos[iatom-3] += gamma*gpos[iatom]
        gpos[iatom-2] += 0.5*(1-gamma)*gpos[iatom]
        gpos[iatom-1] += 0.5*(1-gamma)*gpos[iatom]

if __name__=='__main__':
    # Construction of system and force field including ghost atoms
    file_name = 'init.chk'
    system, ghost_indexes = get_H2O_system(file_name)
    ff_full = ForceField.generate(system, 'pars.txt', rcut=15*angstrom, alpha_scale=3.2, gcut_scale=1.5, smooth_ei=True)

    # Force field where ghosts are not implicitly included, but still taken
    # into account for force calculations
    ff = GhostForceField(ff_full, ghost_indexes, write_ghost_positions_H2O, write_ghost_forces_H2O)
    
    # Tests
    gpos = np.zeros((ff.system.pos.shape[0],ff.system.pos.shape[1]))
    gpos_full = np.zeros((ff_full.system.pos.shape[0],ff_full.system.pos.shape[1]))
    e = ff.compute(gpos=gpos)
    e_full = ff_full.compute(gpos=gpos_full)
    assert e==e_full
#    print np.sum(gpos, axis=0)
#    print np.sum(gpos_full, axis=0)
    assert np.all(np.abs(np.sum(gpos, axis=0)) < 1e-12)
    newpos = ff.system.pos.copy()*(1.0+np.random.normal(0.0,0.01*angstrom,(ff.system.natom,3)))
    ff.update_pos(newpos)
    gpos[:] = 0.0
    gpos_full[:] = 0.0
    e = ff.compute(gpos=gpos)
    e_full = ff_full.compute(gpos=gpos_full)
    assert e==e_full
    assert np.all(np.abs(np.sum(gpos, axis=0)) < 1e-12)
    print e
