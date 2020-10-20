import scipy.linalg
import numpy as np


# from https://github.com/scipy/scipy/pull/3556/files

def funm_psd(A, func, check_finite=True):
    """
    Evaluate a matrix function of a positive semi-definite matrix.
    Returns the value of matrix-valued function ``f`` at `A`. The
    function ``f`` is an extension of the scalar-valued function `func`
    to matrices.
    Parameters
    ----------
    A : (N, N) array_like
        Positive semi-definite matrix.
    func : callable
        Callable object that evaluates a scalar function f.
        Must be vectorized (eg. using vectorize).
    check_finite : boolean, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    ret : (N, N) ndarray
        Value of the matrix function at `A`.

    See also
    --------
    funm : Evaluate a matrix function without the psd restriction.
    Examples
    --------
    >>> from scipy import linalg
    >>> a = np.array([[1, 2], [2, 4]])
    >>> r = linalg.funm_psd(a, np.sqrt)
    >>> r
    array([[ 0.4472136 ,  0.89442719],
           [ 0.89442719,  1.78885438]])
    >>> r.dot(r)
    array([[ 1.,  2.],
           [ 2.,  4.]])
    """
    A = np.asarray(A)
    if len(A.shape) != 2:
        raise ValueError("Non-matrix input to matrix function.")
    w, v = scipy.linalg.eigh(A, check_finite=check_finite)
    w = np.maximum(w, 0)
    return (v * func(w)).dot(v.conj().T)


def sqrtm_psd(A, check_finite=True):
    """
    Matrix square root of a positive semi-definite matrix.
    Parameters
    ----------
    A : (N, N) array_like
        Positive semi-definite matrix whose square root to evaluate.
    check_finite : boolean, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    sqrtm : (N, N) ndarray
        Value of the sqrt function at `A`.

    See also
    --------
    sqrtm : Matrix square root without the psd restriction.
    Examples
    --------
    >>> from scipy import linalg
    >>> a = np.array([[1, 2], [2, 4]])
    >>> r = scipy.linalg.sqrtm_psd(a)
    >>> r
    array([[ 0.4472136 ,  0.89442719],
           [ 0.89442719,  1.78885438]])
    >>> r.dot(r)
    array([[ 1.,  2.],
           [ 2.,  4.]])
    """
    return funm_psd(A, np.sqrt, check_finite)
