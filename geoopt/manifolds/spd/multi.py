import torch


def multitrans(X):
    r"""Returns the tranpose of matrices stacked in an (...,n,n)-shaped array.
    """
    # TODO(ccruceru): Make sure this is not slower than
    #       torch.transpose(X, (-2, -1))
    return torch.einsum("...ji", X)


def multisym(X):
    r"""Returns the symmetrized version of X."""
    return 0.5 * (X + multitrans(X))


def multihgie(W, V):
    r"""The inverse of :py:`torch.symeig` for stacked matrices. The name "hgie"
    is simply the string "eigh" reversed.
    """
    return torch.einsum("...ij,...j,...kj->...ik", V, W, V)


def multisymapply(X, f, *, wmin=None, wmax=None):
    r"""Template function acting on stacked symmetric matrices that applies a
    given analytic function on them via eigenvalue decomposition.
    """
    assert torch.allclose(X, multitrans(X))
    W, V = torch.symeig(X, eigenvectors=True)
    if wmin or wmax:
        W.data.clamp_(min=wmin, max=wmax)
    W = f(W)
    X_new = multihgie(W, V)

    return X_new


def multispdproj(X, *, wmin=None, wmax=None):
    r"""Projects a batch of matrices onto the space of SPD matrices."""
    if not wmin and not wmax:
        raise ValueError("At least one of `wmin` and `wmax` must be given.")

    return multisymapply(multisym(X), lambda W: W, wmin=wmin, wmax=wmax)


def multilog(X, *, wmin=None, wmax=None):
    r"""Computes the matrix-logarithm of several positive definite matrices at
    once.
    """
    return multisymapply(X, torch.log, wmin=wmin, wmax=wmax)


def multiexp(X, *, wmin=None, wmax=None):
    r"""Computes the matrix-exponential of several symmetric matrices at once.
    """
    return multisymapply(X, torch.exp, wmin=wmin, wmax=wmax)


def multisqrt(X, *, wmin=None, wmax=None):
    r"""Computes the matrix square root of several positive definite matrices at
    once.
    """
    return multisymapply(X, torch.sqrt, wmin=wmin, wmax=wmax)


def multiAXAt(A, X):
    r"""Computes the product :math:`A X A^\top` for several matrices at once."""
    return torch.einsum("...ij,...jk,...lk->...il", A, X, A)
