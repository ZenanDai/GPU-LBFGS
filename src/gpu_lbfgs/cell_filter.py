"""Coordinates and forces of ase.filters.FrechetCellFilter (zero pressure, full cell), batched.

For a cell deformation F relative to the initial cell (cell = cell0 @ F.T), the filter uses
    positions part: q = r @ inv(F.T)            (r: Cartesian positions)
    cell part:      N * log(F)                  (N: number of atoms, ASE's default exp_cell_factor)
and the matching generalized forces below.
"""
import torch


@torch.no_grad()
def generalized_forces(logf, deformation, cell, forces, stress, counts):
    """FrechetCellFilter forces for a batch.

    logf, deformation: [B, 3, 3] log(F) and F; cell: [B, 3, 3] current cell (rows = vectors);
    forces: [B, rows, 3] padded atomic forces; stress: [B, 6] (ASE Voigt order xx yy zz yz xz xy)
    or [B, 3, 3]; counts: [B] number of atoms.
    Returns (atom_force [B, rows, 3], cell_force [B, 3, 3], residual [B]), where residual is the
    largest row norm over both parts, the quantity ASE compares with fmax.

    The cell force needs the adjoint of the matrix exponential's derivative at log(F). For the
    Frobenius inner product it equals the derivative at log(F).T applied to the virial term, which
    one 6x6 block exponential gives for each structure.
    """
    if stress.ndim == 2:
        stress = stress[:, [0, 5, 4, 5, 1, 3, 4, 3, 2]].reshape(-1, 3, 3)
    virial = -torch.linalg.det(cell)[:, None, None] * stress
    g = torch.linalg.solve(deformation, virial.transpose(-1, -2)).transpose(-1, -2)   # virial @ inv(F.T)
    block = logf.new_zeros((len(logf), 6, 6))
    block[:, :3, :3] = logf.transpose(-1, -2)
    block[:, 3:, 3:] = logf.transpose(-1, -2)
    block[:, :3, 3:] = g
    cell_force = torch.linalg.matrix_exp(block)[:, :3, 3:] / counts[:, None, None]
    atom_force = forces @ deformation
    residual = torch.maximum(torch.linalg.vector_norm(atom_force, dim=-1).amax(dim=-1),
                             torch.linalg.vector_norm(cell_force, dim=-1).amax(dim=-1))
    return atom_force, cell_force, residual
