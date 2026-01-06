import numpy as np
import random
import psutil
import scipy
from scipy.special import gamma, gammaincinv
from scipy.interpolate import pchip_interpolate
from scipy.signal import lfilter, filtfilt
import traceback
import time
import functools
from scipy import signal
from numpy.linalg import pinv
from numpy import linalg
from scipy.linalg import toeplitz
from scipy.spatial.distance import cdist, euclidean
from numba import jit, njit, cuda
import numba

# Global cache dictionary
mc = {}


@jit(target_backend='cuda', nopython=True, cache=True)  
def block_geometric_median_realtime(X, blocksize=1, tol=1e-5, y=None, max_iter=500):
    if blocksize > 1:
        o, v = X.shape  # #observations & #variables
        r = o % blocksize  # #rest in last block
        b = (o-r) // blocksize  # #blocks
        
        if r > 0:
            X = np.vstack((np.sum(X[:o-r].reshape(blocksize, b*v), axis=0).reshape(b, v), np.sum(X[o-r:], axis=0) * (blocksize/r)))
        else:
            X = np.sum(X.reshape(blocksize, b*v), axis=0).reshape(b, v)
    
    y = geometric_median_realtime(X, tol, y, max_iter)
    
    return y / blocksize

@jit(target_backend='cuda', nopython=True, cache=True)  
def geometric_median_realtime(X, tol=1e-5, y=None, max_iter=500):
    if y is None:
        y = np.median(X, axis=0)
        
    for i in range(max_iter):
        invnorms = 1 / np.sqrt(np.sum((X - y)**2, axis=1))
        oldy = y
        y = np.sum(X * invnorms[:, None], axis=0) / np.sum(invnorms)
        
        if np.linalg.norm(y - oldy) / np.linalg.norm(y) < tol:
            break
            
    return y


def hlp_split(string, delims):
    """
    Split a string according to some delimiter(s).
    
    Parameters:
        string (str): The input string to be split.
        delims (str): A string containing delimiter characters.
    
    Returns:
        result (list): A list of non-empty, non-delimiter substrings in string.
        
    Examples:
        # split a string at colons and semicolons; returns a list of four parts
        print(hlp_split('sdfdf:sdfsdf;sfdsf;;:sdfsdf:', ':;'))
    """
    
    # Find the positions where the string should be split
    pos = [0] + [i + 1 for i, c in enumerate(string) if c in delims] + [len(string) + 1]
    
    # Create the result list by extracting the substrings from the original string
    result = [string[pos[i]: pos[i + 1] - 1] for i in range(len(pos) - 1)]
    
    # Filter out the empty strings
    result = [s for s in result if s]
    
    return result

@jit(target_backend='cuda', nopython=True, cache=True)  
def hlp_memfree_realtime():
    """
    Get the amount of free physical memory, in bytes.
    """
    try:
        # Get the amount of free physical memory in bytes
        result = psutil.virtual_memory().available
    except:
        # If unable to fetch memory details, default to 1,000,000 bytes
        result = 1000000
        
    return result


@jit(target_backend='cuda', nopython=True, cache=True)  
def design_yulewalk_filter_realtime(srate, ab=None):
    if ab is None:
        F = np.array([0, 2, 3, 13, 16, 40, np.minimum(
            80.0, (srate / 2.0) - 1.0), srate / 2.0]) * 2.0 / srate
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = yulewalk(8, F, M)
    else:
        A, B = ab
    
    return B, A

@jit(target_backend='cuda', nopython=True, cache=True)  
def yulewalk(order, F, M):
    """Recursive filter design using a least-squares method.

    [B,A] = YULEWALK(N,F,M) finds the N-th order recursive filter
    coefficients B and A such that the filter:

    B(z)   b(1) + b(2)z^-1 + .... + b(n)z^-(n-1)
    ---- = -------------------------------------
    A(z)    1   + a(1)z^-1 + .... + a(n)z^-(n-1)

    matches the magnitude frequency response given by vectors F and M.

    The YULEWALK function performs a least squares fit in the time domain. The
    denominator coefficients {a(1),...,a(NA)} are computed by the so called
    "modified Yule Walker" equations, using NR correlation coefficients
    computed by inverse Fourier transformation of the specified frequency
    response H.

    The numerator is computed by a four step procedure. First, a numerator
    polynomial corresponding to an additive decomposition of the power
    frequency response is computed. Next, the complete frequency response
    corresponding to the numerator and denominator polynomials is evaluated.
    Then a spectral factorization technique is used to obtain the impulse
    response of the filter. Finally, the numerator polynomial is obtained by a
    least squares fit to this impulse response. For a more detailed explanation
    of the algorithm see [1]_.

    Parameters
    ----------
    order : int
        Filter order.
    F : array
        Normalised frequency breakpoints for the filter. The frequencies in F
        must be between 0.0 and 1.0, with 1.0 corresponding to half the sample
        rate. They must be in increasing order and start with 0.0 and end with
        1.0.
    M : array
        Magnitude breakpoints for the filter such that PLOT(F,M) would show a
        plot of the desired frequency response.

    References
    ----------
    .. [1] B. Friedlander and B. Porat, "The Modified Yule-Walker Method of
           ARMA Spectral Estimation," IEEE Transactions on Aerospace Electronic
           Systems, Vol. AES-20, No. 2, pp. 158-173, March 1984.

    Examples
    --------
    Design an 8th-order lowpass filter and overplot the desired
    frequency response with the actual frequency response:

    >>> f = [0, .6, .6, 1]         # Frequency breakpoints
    >>> m = [1, 1, 0, 0]           # Magnitude breakpoints
    >>> [b, a] = yulewalk(8, f, m) # Filter design using a least-squares method

    """
    F = np.asarray(F)
    M = np.asarray(M)
    npt = 512
    lap = np.floor(npt / 25 + 0.5)
    lap = int(lap)
    mf = F.size
    npt = npt + 1  # For [dc 1 2 ... nyquist].
    Ht = np.zeros((1, npt))
    Ht = Ht[0]
    nint = mf - 1
    df = np.diff(F)

    nb = 0
    Ht[0] = M[0]

    for i in range(nint):
        if df[i] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.floor(F[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)

        for index, j_value in enumerate(j):
            if ne == nb:
                inc = 0
            else:
                inc = (j_value - nb) / (ne - nb)
            Ht[j_value] = inc * M[i + 1] + (1 - inc) * M[i]

        nb = ne + 1

    Ht = np.hstack((Ht, Ht[-2:0:-1]))
    n = Ht.size
    n2 = np.floor((n + 1) / 2)
    nb = order
    nr = 4 * order
    nt = np.arange(0, nr)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[0:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))   # pick NR correlations  # noqa

    # Form window to be used in extracting the right "wing" of two-sided
    # covariance sequence
    Rwindow = np.concatenate(
        (np.array([[1 / 2]]), np.ones((1, int(n2 - 1))), np.zeros((1, int(n - n2)))),
        axis=1)
    A = polystab(denf(R, order))  # compute denominator

    # compute additive decomposition
    Qh = numf(np.concatenate((R[0] / 2, R[1:nr]), axis=None), A, order)

    # compute impulse response
    _, Ss = 2 * np.real(signal.freqz(Qh, A, worN=n, whole=True))

    hh = np.fft.ifft(
        np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss, dtype=complex))))
    )
    B = np.real(numf(hh[0:nr], A, nb))

    return B, A


@jit(target_backend='cuda', nopython=True, cache=True)  
def yulewalk_filter_realtime(X, B, A, zi=None, axis=-1):
    # apply the signal shaping filter and initialize the IIR filter state
    if zi is None:
        # zi = signal.lfilter_zi(B, A)
        # zi = np.transpose(X[:, 0] * zi[:, None])
        # Init zero state to mimic matlab, 
        zi = np.zeros((X.shape[0], X.shape[0] + 1))
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)
    else:
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)

    return out, zf

@jit(target_backend='cuda', nopython=True, cache=True)  
def compute_histogram(x, bins):
    return np.histogram(x, bins=bins)[0]

@jit(target_backend='cuda', nopython=True, cache=True)  
def count_greater_than_threshold(x, threshold):
    return np.sum(x >= threshold)

import numba
from numba_scipy import special
import scipy

# import numba_special
@jit(target_backend='cuda', nopython=True, cache=True)  
def fit_eeg_distribution_realtime(X, min_clean_fraction=0.25, max_dropout_fraction=0.1, quants=np.array([0.022, 0.6]), step_sizes=np.array([0.01, 0.01]), beta_range=np.linspace(1.7, 3.5, 13)):
    # sort data so we can access quantiles directly
    X = np.array(X).ravel()
    X = np.sort(X)
    n = len(X)
    
    # preallocate arrays for zbounds and rescale
    zbounds = np.empty((len(beta_range), len(quants)))
    rescale = np.empty(len(beta_range))
        
    for idx, b in enumerate(beta_range):
        quant_sign_diff = np.sign(quants - 0.5)
        gammaincinv_2d = quant_sign_diff * (2 * quants - 1)
        gammaincinv_final = np.empty(gammaincinv_2d.shape[0])
        for i, y in enumerate(gammaincinv_2d):
            gammaincinv_final[i] = scipy.special.gammaincinv(1/b, y)
        
        zbounds[idx, :] = quant_sign_diff * gammaincinv_final ** (1/b)
        
        # For the gamma function within the rescale formula, you should use the provided special functions from SciPy or NumPy.
        # Note that `sc.gamma` and `np.gamma` are equivalent.
        rescale[idx] = b/(2 * scipy.special.gamma(1/b))
        
    # determine the quantile-dependent limits for the grid search
    lower_min = min(quants)
    max_width = np.diff(quants)[0]
    min_width = min_clean_fraction * max_width
    
    # get matrix of shifted data ranges
    # Compute the indices    
    start_arrange = int(np.round(n*max_width))
    start_arrange = np.round(np.arange(1, start_arrange + 1))
    for i in range(len(start_arrange)):
        start_arrange[i] = int(start_arrange[i])
    start_idx = start_arrange
    
    end_arrange = np.arange(lower_min, lower_min + max_dropout_fraction + step_sizes[0]*0.5, step_sizes[0])
    end_arrange = np.round(n * end_arrange)
    for i in range(len(end_arrange)):
        end_arrange[i] = int(end_arrange[i])
    end_idx = end_arrange

    # Use the computed indices to slice the array
    # X_range = np.array([X[idx + end_idx - 1] for idx in start_idx])

    X_range = np.empty((len(start_idx), len(X)))
    for i in range(len(start_idx)):
        for j in range(len(X)):
            X_range[i, j] = X[int(j + start_idx[i] + end_idx[i] - 1)]


    X1 = X_range[0, :]
    X_range = X_range - X1

    opt_val = np.inf
    # for each interval width...
    ms = np.round(n * np.arange(max_width, min_width, -step_sizes[1]))
    for i in range(len(ms)):
        ms[i] = int(ms[i])

    for m in ms:
        m = int(m)
        # scale and bin the data in the intervals
        nbins = int(np.round(3 * np.log2(1 + m/2)))
        H = X_range[:m] 
        H *= (nbins)
        H /= X_range[m - 1]
        
        # Define bins for the histogram
        bins = np.arange(nbins+1)
        # hist_counts = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], 0, H)
        hist_counts = np.empty((len(bins) - 1, H.shape[1]))  # Assuming H.shape[1] is the dimension along which you apply the function
        inf_bin_counts = np.empty(H.shape[1])

        threshold = nbins-1

        for i in range(H.shape[1]):
            hist_counts[:, i] = compute_histogram(H[:, i], bins=bins)
            inf_bin_counts[i] = count_greater_than_threshold(H[:, i], threshold)

        
        hist_counts = np.vstack((hist_counts, inf_bin_counts[np.newaxis, :]))
        logq = np.log(hist_counts + 0.01)
     
        # for each shape value...
        for k, b in enumerate(beta_range):         
            bounds = zbounds[k]
            
            # evaluate truncated generalized Gaussian pdf at bin centers
            x = bounds[0] + (0.5 + np.arange(0, nbins)) / nbins * np.diff(bounds)
            p = np.exp(-np.abs(x)**b) * rescale[k]
            p = p / np.sum(p)
            
            # calculate KL divergences
            kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m)
                        
            # update optimal parameters
            min_val = np.min(kl)
            idx = np.argmin(kl)
            
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = b
                opt_bounds = bounds  
                idx = int(idx) 
                opt_lu_1 = X1[idx]
                opt_lu_2 = X1[idx] + X_range[m-1, idx]
 
    # recover distribution parameters at optimum
    alpha = (opt_lu_2 - opt_lu_1) / np.diff(opt_bounds)
    # print(np.array(opt_lu).shape)
    mu = opt_lu_1 - opt_bounds[0] * alpha
    beta = opt_beta
    
    # calculate the distribution's standard deviation from alpha and beta
    sig = np.sqrt((alpha**2) * scipy.special.gamma(3/beta) / scipy.special.gamma(1/beta))

    return mu, sig, alpha, beta




@jit(target_backend='cuda', nopython=True, cache=True)  
def moving_average_realtime(N, X, Zi=None):
    """
    Run a moving-average filter along the second dimension of the data.
    """
    if Zi is None:
        Zi = np.zeros((X.shape[0], N))
        
    Y = np.concatenate((Zi, X), axis=1)
    M = Y.shape[1]
    I = np.array([np.arange(1, M - N + 1), np.arange(1 + N, M + 1)])
    S = np.array([-np.ones(M - N), np.ones(M - N)]) / N
    
    I = I - 1
    X = np.cumsum(Y[:, I.ravel(order='F')] * S.ravel(order='F'), axis=1)

    X = X[:, 1::2]

    Zf = np.concatenate((-(X[:, -1:] * N - Y[:, -N:-N + 1]), Y[:, -N + 1:]), axis=1)
    
    return X, Zf


@jit(target_backend='cuda', nopython=True, cache=True)  
def polystab(a):
    """Polynomial stabilization.

    POLYSTAB(A), where A is a vector of polynomial coefficients,
    stabilizes the polynomial with respect to the unit circle;
    roots whose magnitudes are greater than one are reflected
    inside the unit circle.

    Examples
    --------
    Convert a linear-phase filter into a minimum-phase filter with the same
    magnitude response.

    >>> h = fir1(25,0.4);               # Window-based FIR filter design
    >>> flag_linphase = islinphase(h)   # Determines if filter is linear phase
    >>> hmin = polystab(h) * norm(h)/norm(polystab(h));
    >>> flag_minphase = isminphase(hmin)# Determines if filter is minimum phase

    """
    v = np.roots(a)
    i = np.where(v != 0)
    vs = 0.5 * (np.sign(np.abs(v[i]) - 1) + 1)
    v[i] = (1 - vs) * v[i] + vs / np.conj(v[i])
    ind = np.where(a != 0)
    b = a[ind[0][0]] * np.poly(v)

    # Return only real coefficients if input was real:
    if not np.sum(np.imag(a)):
        b = np.real(b)

    return b

@jit(target_backend='cuda', nopython=True, cache=True)  
def numf(h, a, nb):
    """Find numerator B given impulse-response h of B/A and denominator A.

    NB is the numerator order.  This function is used by YULEWALK.
    """
    nh = np.max(h.size)
    xn = np.concatenate((1, np.zeros((1, nh - 1))), axis=None)
    impr = signal.lfilter(np.array([1.0]), a, xn)

    b = linalg.lstsq(
        toeplitz(impr, np.concatenate((1, np.zeros((1, nb))), axis=None)),
        h.T, rcond=None)[0].T

    return b

@jit(target_backend='cuda', nopython=True, cache=True)  
def denf(R, na):
    """Compute denominator from covariances.

    A = DENF(R,NA) computes order NA denominator A from covariances
    R(0)...R(nr) using the Modified Yule-Walker method. This function is used
    by YULEWALK.

    """
    nr = len(R)
    Rm = scipy.linalg.toeplitz(R[na:nr - 1], R[na:0:-1])
    Rhs = - R[na + 1:nr]
    A = np.concatenate(
        (1, linalg.lstsq(Rm, Rhs.T, rcond=None)[0].T), axis=None)
    return A