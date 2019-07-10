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


def multisymapply(X, f, posdef=False, eps=1e-8):
    r"""Template function acting on stacked symmetric matrices that applies a
    given analytic function on them via eigenvalue decomposition.
    """
    assert torch.allclose(X, multitrans(X))
    W, V = torch.symeig(X, eigenvectors=True)
    if posdef:
        W.data.clamp_(min=eps)
    W = f(W)
    X_new = multihgie(W, V)

    return X_new


def multilog(X, eps=1e-8):
    r"""Computes the matrix-logarithm of several positive definite matrices at
    once.
    """
    return multisymapply(X, torch.log, posdef=True, eps=eps)


def multiexp(X):
    r"""Computes the matrix-exponential of several symmetric matrices at once.
    """
    return multisymapply(X, torch.exp, posdef=False)


def multisqrt(X, eps=1e-8):
    r"""Computes the matrix square root of several positive definite matrices at
    once.
    """
    return multisymapply(X, torch.sqrt, posdef=True, eps=eps)


def multiAXAt(A, X):
    r"""Computes the product :math:`A X A^\top` for several matrices at once."""
    return torch.einsum("...ij,...jk,...lk->...il", A, X, A)
