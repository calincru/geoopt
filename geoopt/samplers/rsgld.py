import math

import torch

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import R
from geoopt.samplers.base import Sampler
from ..utils import copy_or_set_

__all__ = ["RSGLD"]


class RSGLD(Sampler):
    r"""Riemannian Stochastic Gradient Langevin Dynamics

    Parameters
    ----------
    params : iterable
        iterables of tensors for which to perform sampling
    epsilon : float
        step size
    """

    def __init__(self, params, epsilon=1e-3):
        defaults = dict(epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure):
        """Performs a single sampling step.

        Arguments
        ---------
        closure: callable
            A closure that reevaluates the model
            and returns the log probability.
        """
        logp = closure()
        logp.backward()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        manifold = p.manifold
                    else:
                        manifold = R()

                    egrad2rgrad, retr = manifold.egrad2rgrad, manifold.retr
                    epsilon = group["epsilon"]

                    n = torch.randn_like(p).mul_(math.sqrt(epsilon))
                    r = egrad2rgrad(p, 0.5 * epsilon * p.grad + n)
                    # use copy only for user facing point
                    copy_or_set_(p, retr(p, r))
                    p.grad.zero_()

        if not self.burnin:
            self.steps += 1
            self.log_probs.append(logp.item())

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons
        """
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        continue
                    copy_or_set_(p, p.manifold.projx(p))
