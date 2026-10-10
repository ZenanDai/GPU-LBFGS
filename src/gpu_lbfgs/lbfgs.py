"""The L-BFGS update of ase.optimize.LBFGS, batched over structures.

Coordinates and forces are tensors of shape [B, rows, 3] (B structures, rows padded to a common
length with zeros). Padded rows stay zero in every quantity, so they never contribute to inner
products or to the step-length cap.
"""
import torch


def batch_dot(a, b):
    """Inner product per structure of two [B, rows, 3] tensors -> [B]."""
    return (a * b).sum(dim=(-1, -2))


@torch.no_grad()
def two_loop_direction(forces, s_hist, y_hist, rho_hist, h0):
    """Search direction p = -H g from the L-BFGS two-loop recursion (g = -forces)."""
    q = -forces
    a = []
    for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
        ai = rho * batch_dot(s, q)
        q = q - ai[:, None, None] * y
        a.append(ai)
    z = h0 * q
    for (s, y, rho), ai in zip(zip(s_hist, y_hist, rho_hist), reversed(a)):
        b = rho * batch_dot(y, z)
        z = z + s * (ai - b)[:, None, None]
    return -z


@torch.no_grad()
def cap_step(direction, maxstep):
    """ASE LBFGS.determine_step: if the longest row is longer than maxstep, scale it to maxstep."""
    longest = torch.linalg.vector_norm(direction, dim=-1).amax(dim=-1)
    scale = torch.where(longest >= maxstep, maxstep / longest.clamp_min(1e-300), torch.ones_like(longest))
    return direction * scale[:, None, None]


class BatchLBFGS:
    """ase.optimize.LBFGS (no line search) for a batch of structures.

    alpha: initial inverse Hessian 1/alpha; memory: number of stored (s, y) pairs;
    maxstep: largest row displacement per step; damping: factor applied to every step.
    The defaults are those of ASE.
    """

    def __init__(self, maxstep=0.2, memory=100, alpha=70.0, damping=1.0):
        if not (maxstep > 0 and alpha > 0 and damping > 0) or memory < 1:
            raise ValueError("maxstep, alpha and damping must be positive and memory >= 1")
        self.maxstep, self.memory, self.h0, self.damping = maxstep, memory, 1.0 / alpha, damping
        self.s, self.y, self.rho = [], [], []
        self.x0 = self.f0 = None

    @torch.no_grad()
    def step(self, x, f):
        """New coordinates from the current coordinates x and forces f (ASE LBFGS.step)."""
        if self.x0 is not None:                     # ASE LBFGS.update
            s, y = x - self.x0, self.f0 - f
            self.s.append(s)
            self.y.append(y)
            self.rho.append(1.0 / batch_dot(y, s))
            if len(self.s) > self.memory:
                self.s.pop(0), self.y.pop(0), self.rho.pop(0)
        direction = two_loop_direction(f, self.s, self.y, self.rho, self.h0)
        self.x0, self.f0 = x, f
        return x + cap_step(direction, self.maxstep) * self.damping

    def keep(self, index):
        """Keep only the structures at `index` (a 1-D LongTensor), e.g. after some converged."""
        pick = lambda t: t.index_select(0, index)
        self.s, self.y, self.rho = [pick(t) for t in self.s], [pick(t) for t in self.y], [pick(t) for t in self.rho]
        if self.x0 is not None:
            self.x0, self.f0 = pick(self.x0), pick(self.f0)
