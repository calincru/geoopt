import torch
from ..base import Manifold
from .multi import *
import geoopt

__all__ = ["SymmetricPositiveDefinite"]


class SymmetricPositiveDefinite(Manifold):
    r"""The manifold of symmetric positive definite matrices."""

    name = "Symmetrc Positive Definite"
    ndim = 2
    reversible = False

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        ok = torch.allclose(x, multitrans(x), atol=atol, rtol=rtol)
        if not ok:
            return (
                False,
                "The matrix is not symmetric with atol={}, rtol={}".format(atol, rtols),
            )
        w, _ = torch.symeig(x)
        if not all(w > atol):
            return (
                False,
                "The matrix is not positive definite with atol={}".format(atol),
            )
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        ok = torch.allclose(x, multitrans(x), atol=atol, rtol=rtol)
        if not ok:
            return (
                False,
                "The matrix is not symmetric with atol={}, rtol={}".format(atol, rtols),
            )
        return True, None

    def inner(self, x, u, v=None, *, keepdim=False):
        x_inv_u = torch.solve(x, u)[0]
        if v is None:
            x_inv_v = x_inv_u
        else:
            x_inv_v = torch.solve(x, v)[0]
        return torch.matmul(x_inv_u, x_inv_v).sum((-2, -1), keepdim=keepdim)

    # TODO(ccruceru): Maybe use the alternative implementation of the norm if
    # the solve() above really proves problematic (see pymanopt).

    def proju(self, x, u):
        return 0.5 * (u + multitrans(u))

    def projx(self, x):
        # symmetrize it and then clamp its eigenvalues
        return multisymapply(multisym(x), lambda W: torch.clamp(W, min=0))

    def expmap(self, x, u):
        l = torch.cholesky(x)
        l_inv = torch.inverse(l)
        a = multiAXAt(l_inv, u)
        expa = multiexp(a)
        expx_y = multiAXAt(l, expa)

        return expx_y

    def logmap(self, x, y):
        l = torch.cholesky(x)
        l_inv = torch.inverse(l)
        a = multiAXAt(l_inv, y)
        loga = multilog(a)
        logx_y = multiAXAt(l, loga)

        return logx_y

    def retr(self, x, u):
        # TODO(ccruceru): Maybe symmetrize for numerical stability.
        return x + u + 0.5 * torch.matmul(u, torch.solve(u, x)[0])

    def dist(self, x, y, *, keepdim=False, squared=False):
        l = torch.cholesky(x)
        l_inv = torch.cholesky(l)
        a = multiAXAt(l_inv, y_inv)
        loga = multilog(a)
        sq_dist = loga.pow(2).sum((-2, -1), keepdim=keepdim)  # batched trace

        return sq_dist if squared else torch.sqrt(sq_dist)

    # TODO(ccruceru): Ignoring the :py:`more` args as I'm still note sure what
    # the convention is about applying the member functions on stacked values.

    def transp(self, x, y, v, *more):
        return v

    def ptransp(self, x, y, v):
        r"""The actual parallel transport. Much more expensive, and it seems
        that most other libraries use the one from above.
        """
        y_x_inv = multitrans(torch.solve(multitrans(y), multitrans(x))[0])  # y/x
        l = torch.cholesky(y_x_inv)
        vs = multiAXAt(l, v)

        return vs

    # TODO(ccruceru): Consider if the combined {expmap,retr}_transp methods do
    # improve if we instead use the real parallel transport function.

    def egrad2rgrad(self, x, u):
        return multiAXAt(x, multisym(u))
