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

    def __init__(self, wmin=1e-8, wmax=1e8, requires_grad=False):
        self.wmin = wmin
        self.wmax = wmax
        self.requires_grad = requires_grad

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        ok = torch.allclose(x, multitrans(x), atol=atol, rtol=rtol)
        if not ok:
            return (
                False,
                "The matrix is not symmetric with atol={}, rtol={}".format(atol, rtols),
            )
        w, _ = torch.symeig(x)
        if not all(w > self.wmin):
            return (
                False,
                "The matrix is not positive definite with atol={}".format(self.wmin),
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

    def _inner_no_grad(self, x, u, v=None, *, keepdim=False):
        l = torch.cholesky(x)
        x_inv_u = torch.cholesky_solve(u, l)
        if v is None:
            x_inv_v = x_inv_u
        else:
            x_inv_v = torch.cholesky_solve(v, l)
        return torch.matmul(x_inv_u, x_inv_v).sum((-2, -1), keepdim=keepdim)

    def _inner(self, x, u, v=None, *, keepdim=False):
        # TODO(ccruceru): Get rid of it once the derivative of
        # `torch.cholesky_solve` is implemented.
        x_inv_u = torch.solve(u, x)[0]
        if v is None:
            x_inv_v = x_inv_u
        else:
            x_inv_v = torch.solve(v, x)[0]
        return torch.matmul(x_inv_u, x_inv_v).sum((-2, -1), keepdim=keepdim)

    def inner(self, x, u, v=None, *, keepdim=False):
        inner_fn = self._inner if self.requires_grad else self._inner_no_grad
        return inner_fn(x, u, v=v, keepdim=keepdim)

    # TODO(ccruceru): Maybe use the alternative implementation of the norm if
    # the solve() above really proves problematic (see pymanopt).

    def proju(self, x, u):
        return 0.5 * (u + multitrans(u))

    def projx(self, x):
        # symmetrize it and then clamp its eigenvalues
        return multispdproj(x, wmin=self.wmin, wmax=self.wmax)

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
        # need to compute :math:`X + U + \frac{1}{2} U X^{-1} U`
        # the product is computed as:
        #       U X^{-1} U = U (L L^t)^{-1} U = (L^{-1} U)^t (L^{-1} U)
        l = torch.cholesky(x)
        l_inv_u = torch.triangular_solve(u, l, upper=False)
        y = torch.matmul(multitrans(l_inv_u), l_inv_u)

        return y

    def dist(self, x, y, *, keepdim=False, squared=False):
        l = torch.cholesky(x)
        l_inv = torch.inverse(l)
        a = multiAXAt(l_inv, y)
        w, _ = torch.symeig(a, eigenvectors=self.requires_grad)
        w.data.clamp_(min=self.wmin, max=self.wmax)
        sq_dist = w.log().pow(2).sum(-1, keepdim=keepdim)

        return sq_dist if squared else torch.sqrt(sq_dist)

    # TODO(ccruceru): Ignoring the :py:`more` args as I'm still note sure what
    # the convention is about applying the member functions on stacked values.

    def transp(self, x, y, v, *more):
        return v

    def ptransp(self, x, y, v):
        r"""The actual parallel transport. Use it as follows in optimizers:

        .. py::
            spd = geoopt.SymmetricPositiveDefinite()
            spd.transp = spd.ptransp
        """
        # TODO(ccruceru): Requires matrix square root. See
        # https://github.com/pytorch/pytorch/issues/9983.
        # TODO(ccruceru): Consider if the combined {expmap,retr}_transp methods
        # do improve if we instead use the real parallel transport function.
        raise NotImplementedError

    def egrad2rgrad(self, x, u):
        return multiAXAt(x, multisym(u))
