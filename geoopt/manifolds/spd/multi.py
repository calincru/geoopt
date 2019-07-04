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


def _multi_apply_on_sym(X, f):
    r"""Template function acting on stacked symmetric matrices that applies a
    given analytic function on them via eigenvalue decomposition.
    """
    W, V = torch.symeig(X, eigenvectors=True)
    W = f(W)
    X_new = multihgie(W, V)

    return X_new


def multilog(X):
    r"""Computes the matrix-logarithm of several matrices at once."""
    return _multi_apply_on_sym(X, torch.log)


def multiexp(X):
    r"""Computes the matrix-exponential of several matrices at once."""
    return _multi_apply_on_sym(X, torch.exp)


def multiAXAt(A, X):
    r"""Computes the product :math:`A X A^\top` for several matrices at once."""
    return torch.einsum("...ij,...jk,...lk->...il", A, X, A)
