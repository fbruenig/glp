"""Cell-list updates must never silently return a wrong neighbour list.

After allocation the cell list has a fixed per-cell capacity. If atoms later crowd into one cell
beyond that capacity, update_fn must flag it (Neighbors.overflow) so the caller reallocates;
without the flag the returned list misses (and misplaces) pairs and forces are silently wrong.
"""
from unittest import TestCase

import numpy as np
import jax
import jax.numpy as jnp
from ase.build import bulk
from ase.neighborlist import neighbor_list as ase_neighbor_list

from glp.neighborlist import quadratic_neighbor_list

CUTOFF, LR_CUTOFF, SKIN = 4.5, 6.0, 0.0


def allocate_and_update():
    atoms = bulk("Ar", cubic=True) * [5, 5, 5]          # 500 atoms, 26.3 A box: cell list is used
    # generous pair capacity, so only the cell list (not the pair buffer) can overflow
    allocate, update = quadratic_neighbor_list(
        atoms.get_cell().array, CUTOFF, SKIN, use_cell_list=True, lr_cutoff=LR_CUTOFF,
        capacity_multiplier=3.0,
    )
    neighbors = allocate(jnp.asarray(atoms.get_positions()))   # glp pads with index N: JAX arrays
    assert neighbors.cell_list is not None
    return atoms, neighbors, jax.jit(update)


def pair_set(neighbors, n_atoms):
    c, o = np.asarray(neighbors.centers), np.asarray(neighbors.others)
    m = o < n_atoms
    return set(zip(c[m].tolist(), o[m].tolist()))


def exact_pair_set(atoms, positions):
    a = atoms.copy()
    a.positions = positions
    i, j = ase_neighbor_list("ij", a, CUTOFF)
    return set(zip(i.tolist(), j.tolist()))


class TestCellListOverflow(TestCase):
    def test_small_moves_keep_list_exact(self):
        atoms, neighbors, update = allocate_and_update()
        positions = atoms.get_positions() + np.random.default_rng(0).normal(scale=0.1, size=(len(atoms), 3))
        neighbors = update(jnp.asarray(positions), neighbors)
        assert not bool(neighbors.overflow)
        assert pair_set(neighbors, len(atoms)) == exact_pair_set(atoms, positions)

    def test_overfilled_cell_is_flagged(self):
        atoms, neighbors, update = allocate_and_update()
        capacity = int(neighbors.cell_list.capacity)
        # move the capacity + 4 atoms nearest to one point onto a 1.05 A grid around it (no overlaps)
        positions = atoms.get_positions().copy()
        moved = np.argsort(np.linalg.norm(positions - 3.0, axis=1))[:capacity + 4]
        side = int(np.ceil(len(moved) ** (1 / 3)))
        grid = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)
        positions[moved] = 1.2 + 1.05 * grid[:len(moved)]
        neighbors = update(jnp.asarray(positions), neighbors)
        exact = pair_set(neighbors, len(atoms)) == exact_pair_set(atoms, positions)
        assert bool(neighbors.overflow) or exact, (
            "cell list exceeded its capacity and returned a wrong neighbour list without "
            "setting the overflow flag")
