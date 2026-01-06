import numpy as np
import random
import psutil
import yulewalker as yw
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
from scipy.special import gamma, gammaincinv

# Global cache dictionary
mc = {}


def pop_select(EEG, channels=None, exclude=None):
    """
    Selects or excludes specific channels from EEG data.
    
    Parameters:
        EEG: The input EEG data (numpy array of shape (channels, samples)).
        channels: List of channel indices to select. If None, all channels are selected.
        exclude: List of channel indices to exclude. If None, no channels are excluded.
    
    Returns:
        EEG: The EEG data after selecting or excluding channels.
    """
    if channels is not None:
        EEG = EEG[channels, :]
    if exclude is not None:
        EEG = np.delete(EEG, exclude, axis=0)
    
    return EEG


def block_geometric_median(X, blocksize=1, tol=1e-5, y=None, max_iter=500):
    if blocksize > 1:
        o, v = X.shape  # #observations & #variables
        r = o % blocksize  # #rest in last block
        b = (o-r) // blocksize  # #blocks
        
        if r > 0:
            X = np.vstack((np.sum(X[:o-r].reshape(blocksize, b*v), axis=0).reshape(b, v), np.sum(X[o-r:], axis=0) * (blocksize/r)))
        else:
            X = np.sum(X.reshape(blocksize, b*v), axis=0).reshape(b, v)
    
    y = geometric_median(X, tol, y, max_iter)
    
    return y / blocksize


def geometric_median(X, tol=1e-5, y=None, max_iter=500):
    if y is None:
        y = np.median(X, axis=0)
        
    for i in range(max_iter):
        invnorms = 1 / np.sqrt(np.sum((X - y)**2, axis=1))
        oldy = y
        y = np.sum(X * invnorms[:, None], axis=0) / np.sum(invnorms)
        
        if np.linalg.norm(y - oldy) / np.linalg.norm(y) < tol:
            break
            
    return y


def design_fir(N, F, A, nfft=None, W=None):
    """
    Design an FIR filter using the frequency-sampling method.
    
    The frequency response is interpolated cubically between the specified frequency points.
    
    Parameters:
    N : int
        Order of the filter
    F : array_like
        Vector of frequencies at which amplitudes shall be defined
        (starts with 0 and goes up to 1; try to avoid too sharp transitions)
    A : array_like
        Vector of amplitudes, one value per specified frequency
    nfft : int, optional
        Number of FFT bins to use (default is None)
    W : array_like, optional
        The window function to use (default is None, which uses Hamming window)
        
    Returns:
    B : numpy.ndarray
        Designed filter kernel
    """
    if nfft is None:
        nfft = max(512, 2 ** int(np.ceil(np.log(N) / np.log(2))))
    
    if W is None:
        W = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, N + 1) / N)

    # Calculate interpolated frequency response
    F_interp = pchip_interpolate(np.round(F * nfft), A, np.arange(0, nfft + 1), axis=0)
    
    # Set phase & transform into time domain
    F_phase = F_interp * np.exp(-(0.5 * N) * 1j * np.pi * np.arange(0, nfft + 1) / nfft)
    B = np.real(np.fft.ifft(np.concatenate([F_phase, np.conj(F_phase[-2:0:-1])])))
    
    # Apply window to kernel
    B = B[0:N + 1] * W
    
    return B



def design_kaiser(lo, hi, atten, odd):
    """
    Design a Kaiser window for a low-pass FIR filter
    
    Parameters:
    lo : float
        Normalized lower frequency of the transition band
        
    hi : float
        Normalized upper frequency of the transition band
        
    atten : float
        Stop-band attenuation in dB (-20log10(ratio))
        
    odd : bool
        Whether the length shall be odd
        
    Returns:
    W : numpy.ndarray
        Designed window
    """
    
    # Determine beta of the kaiser window
    if atten < 21:
        beta = 0
    elif atten <= 50:
        beta = 0.5842 * (atten - 21)**0.4 + 0.07886 * (atten - 21)
    else:
        beta = 0.1102 * (atten - 8.7)
        
    # Determine the number of points
    N = np.round((atten - 7.95) / (2 * np.pi * 2.285 * (hi - lo))) + 1
    
    if odd and N % 2 == 0:
        N = N + 1
    
    # Design the window
    W = np.kaiser(N, beta)
    
    return W


def filter_fast(B, A, X, Zi=None, dim=0):
    if Zi is None:
        Zi = []
        
    lenx = X.shape[dim]
    lenb = len(B)
    
    if lenx == 0:
        # empty X
        Zf = Zi
    elif lenb < 256 or lenx < 1024 or lenx <= lenb or lenx * lenb < 4000000 or not np.all(A == 1):
        # use the regular filter
        if Zi:
            X, Zf = lfilter(B, A, X, zi=Zi, axis=dim)
        else:
            print(B.shape)
            # print(A.shape)
            print(X.shape)

            X, _ = lfilter(B, A, X, axis=dim)            
            Zf = None
    else:
        was_single = X.dtype == np.float32
        
        # fftfilt can be used
        if not Zi:
            # no initial conditions to take care of
            if dim != 0:
                X = np.swapaxes(X, 0, dim)
                
            X = oct_fftfilt(B, X)
            Zf = None
            
            if dim != 0:
                X = np.swapaxes(X, 0, dim)
        else:
            # initial conditions available
            if dim != 0:
                X = np.swapaxes(X, 0, dim)
            
            tmp, _ = lfilter(B, 1, X[:len(B), :], zi=Zi, axis=1)
            
            if len(Zi) != 0:
                _, Zf = lfilter(B, 1, X[-len(B)+1:, :], zi=Zi, axis=1)
            else:
                Zf = None
            
            X = oct_fftfilt(B, X)
            # incorporate the piece
            X[:len(B), :] = tmp
            
            if dim != 0:
                X = np.swapaxes(X, 0, dim)
        
        if was_single:
            X = X.astype(np.float32)
    
    return X, Zf


def oct_fftfilt(b, x):
    transpose = (x.shape[0] == 1)
    
    if transpose:
        x = x.T
        
    r_x = x.shape[0]
    l_b = len(b)
    

    N = 2 ** (np.ceil(np.log2(r_x + l_b - 1)))    
    B = np.fft.fft(b, int(N))

    print(x.shape)  # should probably be (16384,) or (16384, 1) if x is a single signal
    print(b.shape)  # should probably be (l_b,) where l_b is the length of the filter
    print(B.shape)  # should be (16384,) after the FFT
    print(B[:, None].shape)  # should be (16384, 1) after adding a new axis

    y = np.fft.ifft(np.fft.fft(x, int(N), axis=0) * B[:, None])

    y = y[:r_x, :]
    
    if transpose:
        y = y.T
    
    if np.isrealobj(b) and np.isrealobj(x):
        y = np.real(y)
        
    return y


def filtfilt_fast(*args):
    """
    Like filtfilt(), but faster when filter and signal are long (and A=1).
    Y = filtfilt_fast(B,A,X)

    Uses FFT convolution (needs fftfilt). The function is faster than filter when approx.
    length(B)>256 and size(X,Dim)>1024, otherwise slower (due size-testing overhead).

    Note:
     Can also be called with four arguments, as Y = filtfilt_fast(N,F,A,X), in which case an Nth order
     FIR filter is designed that has the desired frequency response A at normalized frequencies F; F
     must be a vector of numbers increasing from 0 to 1.

    See also: 
      filtfilt, filter
    """
    if len(args) == 3:
        B, A, X = args
    elif len(args) == 4:
        N, F, M, X = args
        B = design_fir(N, F, np.sqrt(M))
        A = 1
    else:
        print("Invalid number of arguments. See function documentation.")
        return

    if A == 1:
        was_single = X.dtype == np.float32
        w = len(B)
        t = X.shape[0]
        
        # extrapolate
        prepend = 2 * X[0] - X[((w + 1) - np.arange(2, w + 2)) % t]
        # Double the last element of X
        double_last_element = 2 * X[t - 1]
        # Create the vector (from t-1 to t-w in decrements of 1) and adjust to 0-based index
        index_vector = (t - 2 - np.arange(0, w)) % t
        # Select specific elements from X using the above index vector
        selected_elements = X[index_vector]
        # Element-wise subtraction
        append = double_last_element - selected_elements

        X = np.concatenate([prepend, X, append])

        
        # filter, reverse
        X, _ = filter_fast(B, A, X)
        X = X[::-1, :]
        
        print('DBG 1: ', X.shape)
        
        # filter, reverse
        X, _ = filter_fast(B, A, X)
        X = X[::-1, :]
    
        print('DBG 2')

        # remove extrapolated pieces
        X = np.delete(X, np.concatenate((np.arange(0, w), np.arange(t + w, t + 2 * w))), axis=0)
        
        if was_single:
            X = X.astype(np.float32)
    else:
        # fall back to filtfilt for the IIR case
        X = filtfilt(B, A, X)
    
    return X


def hlp_handleerror(e, level=0):
    """
    Displays a formatted error message for some exception object, including a full stack trace.
    
    Parameters:
        e (Exception): The exception object, as received from a catch clause.
        level (int): Optional indentation level, in characters. Defaults to 0.
        
    Returns:
        None: If the function is called with no return value, the error report is printed to the console.
        str: If the function is called with a return value, the error report is returned as a string.
    """
    
    try:
        # Compute the appropriate indentation level
        indent = ' ' * level
        
        # Generate the error message
        messages = str(e).split('\n')
        formatted_messages = [f"{indent} {message}" for message in messages]
        
        # Generate the stack trace
        stack_trace = traceback.extract_tb(e.__traceback__)
        formatted_trace = [f"{indent}   {frame.filename}: {frame.lineno}" for frame in stack_trace]
        
        # Combine messages and stack trace
        full_report = '\n'.join(formatted_messages + [f"{indent}occurred in: "] + formatted_trace)
        
        return full_report

    except Exception as e2:
        return "An error occurred, but the traceback could not be displayed due to another error: {}".format(e2)



def hlp_microcache(domain, func=None, *args, **config_options):
    if func is None:
        # Advanced usage: Clear or reset cache, or set configurations
        handle_advanced_usage(domain, config_options)
    elif callable(func):
        # Regular call: Cache results of functions for repeated calls with the same arguments
        return cache_function_call(domain, func, *args)
    else:
        raise ValueError("Invalid function argument")


def cache_function_call(domain, func, *args):
    now = time.time()
    key = (func, args)
    
    # Check if domain exists in cache, and if not, initialize it
    if domain not in mc:
        mc[domain] = {
            'config': {},
            'cache': {}
        }

    cache = mc[domain]['cache']
    
    if key in cache:
        # Cache hit, update the metadata and return the result
        result, meta = cache[key]
        meta['frequency'] += 1
        meta['last_used'] = now
        return result
    else:
        # Cache miss, compute result, store it in cache, and return it
        result = func(*args)
        meta = {
            'last_used': now,
            'frequency': 1
        }
        cache[key] = (result, meta)
        return result


def handle_advanced_usage(domain, config_options):
    global mc
    
    # Handle 'clear' or 'reset' commands
    if config_options == {}:
        if domain == 'clear':
            mc = {}  # Clear all domains
        elif domain == 'reset':
            mc = {}  # Reset all domains, including configurations
        else:
            # Clear or reset a specific domain
            if domain in mc:
                mc.pop(domain)
    else:
        # Update configurations for a specific domain
        if domain not in mc:
            mc[domain] = {
                'config': {},
                'cache': {}
            }
        mc[domain]['config'].update(config_options)
        


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


def hlp_varargin2struct(args, **varargin):
    """
    Convert a list of name-value pairs into a dictionary with values assigned to names.
    
    Parameters:
    args : list
        List of name-value pairs and/or dictionaries (with values assigned to names).
    
    **varargin : name-value pairs
        Optional name-value pairs, encoding defaults; multiple alternative names may be specified 
        in a list.
    
    Returns:
    result : dict
        A dictionary with fields corresponding to the passed arguments (plus the defaults that were
        not overridden).
    
    Examples:
    args = [{'param1': 100}, 'param2', 200, 'param3', 300]
    result = hlp_varargin2struct(args, param1=10, param2=20, param3='__arg_mandatory__')
    print(result)
    """
    
    # Create the result dictionary from default values
    result = dict(varargin)
    
    # Create a remapping table for alternative default names
    name_for_alternative = {}
    for key, value in varargin.items():
        if isinstance(key, list):
            for alt_name in key[1:]:
                name_for_alternative[alt_name] = key[0]
            result[key[0]] = value
            del result[key]
    
    # Flatten input args if they contain dictionaries
    flat_args = []
    for item in args:
        if isinstance(item, dict):
            for k, v in item.items():
                flat_args.append(k)
                flat_args.append(v)
        else:
            flat_args.append(item)
    
    # Rewrite alternative names into their standard form and override defaults with arguments
    for k in range(0, len(flat_args), 2):
        name = flat_args[k]
        value = flat_args[k + 1]
        
        # Handle alternative names
        if name in name_for_alternative:
            name = name_for_alternative[name]
            
        result[name] = value
    
    # Check for missing but mandatory args
    missing_entries = [k for k, v in result.items() if v == '__arg_mandatory__']
    if missing_entries:
        raise ValueError(f"The parameters {missing_entries} were unspecified but are mandatory.")
    
    return result


def sphericalSplineInterpolate(src, dest, lambda_val=1e-5, order=4, type_='spline', tol=1e-7):
    # Normalize the positions onto the sphere
    src = src / np.linalg.norm(src, axis=0)
    dest = dest / np.linalg.norm(dest, axis=0)
    
    # Calculate the cosine of the angles between electrodes
    cosSS = src.T @ src   # angles between source positions
    cosDS = dest.T @ src  # angles between destination positions
    
    # Compute the interpolation matrix to tolerance tol
    Gss = interpMx(cosSS, order, tol)
    Gds, Hds = interpMx(cosDS, order, tol)
    
    # Include the regularization
    if lambda_val > 0:
        Gss = Gss + lambda_val * np.eye(Gss.shape[0])
    
    # Compute the mapping to the polynomial coefficients space
    muGss = 1.0
    C = np.block([[Gss, muGss * np.ones((Gss.shape[0], 1))], 
                  [muGss * np.ones((1, Gss.shape[1])), 0]])
    iC = pinv(C)
    
    # Compute the mapping from source measurements and positions to destination positions
    if type_.lower() == 'spline':
        W = np.hstack([Gds, np.ones((Gds.shape[0], 1)) * muGss]) @ iC[:, :-1]
    elif type_.lower() == 'slap':
        W = Hds @ iC[:-1, :-1]
    else:
        raise ValueError("Invalid type. Must be 'spline' or 'slap'.")
    
    return W, Gss, Gds, Hds


def interpMx(cosEE, order, tol=1e-10):
    G = np.zeros_like(cosEE)
    H = np.zeros_like(cosEE)
    
    for i in range(cosEE.size):
        x = cosEE.flat[i]
        n = 1
        Pns1 = 1
        Pn = x
        tmp = ((2 * n + 1) * Pn) / ((n * n + n) ** order)
        G.flat[i] = tmp
        H.flat[i] = (n * n + n) * tmp
        oGi = np.inf
        dG = abs(G.flat[i])
        oHi = np.inf
        dH = abs(H.flat[i])
        
        for n in range(2, 501):
            Pns2 = Pns1
            Pns1 = Pn
            Pn = ((2 * n - 1) * x * Pns1 - (n - 1) * Pns2) / n
            oGi = G.flat[i]
            oHi = H.flat[i]
            tmp = ((2 * n + 1) * Pn) / ((n * n + n) ** order)
            G.flat[i] = G.flat[i] + tmp
            H.flat[i] = H.flat[i] + (n * n + n) * tmp
            dG = (abs(oGi - G.flat[i]) + dG) / 2
            dH = (abs(oHi - H.flat[i]) + dH) / 2
            
            if dG < tol and dH < tol:
                break
                
    G = G / (4 * np.pi)
    H = H / (4 * np.pi)
    
    return G, H


# TODO: Implement this function
def window_func():
    pass



def calc_projector(locs, num_samples, subset_size):
    """
    Calculate a bag of reconstruction matrices from random channel subsets
    
    Parameters:
    locs: ndarray
        3D coordinates for channel locations
    num_samples: int
        Number of random samples
    subset_size: int
        Size of each random subset of channels
        
    Returns:
    P: ndarray
        Concatenated reconstruction matrices
    """
    rand_samples = []
    
    for _ in range(num_samples):
        tmp = np.zeros(locs.shape[1])
        subset = randsample(range(locs.shape[1]), subset_size)
        tmp[subset, :] = np.real(sphericalSplineInterpolate(locs[:, subset], locs)).T
        rand_samples.append(tmp)
    
    P = np.hstack(rand_samples)
    
    return P

def randsample(X, num):
    """
    Random sample without replacement
    
    Parameters:
    X: list or ndarray
        Population to sample from
    num: int
        Number of samples to draw without replacement
    
    Returns:
    Y: list
        List of sampled elements
    """
    Y = []
    X = list(X)  # Copy the input to avoid modifying the original list/ndarray
    
    while len(Y) < num:
        pick = int(np.floor(np.random.rand() * len(X)))
        Y.append(X[pick])
        X.pop(pick)
    
    return Y


def mad(X):
    """
    Median Absolute Deviation
    
    Parameters:
    X: ndarray
        Input array
    
    Returns:
    Y: float
        Median absolute deviation of the input
    """
    median_X = np.median(X)
    Y = np.median(np.abs(X - median_X))
    
    return Y


def hlp_memfree():
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


def design_yulewalk_filter(srate, ab=None):
    if ab is None:
        F = np.array([0, 2, 3, 13, 16, 40, np.minimum(
            80.0, (srate / 2.0) - 1.0), srate / 2.0]) * 2.0 / srate
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = yulewalk(8, F, M)
    else:
        A, B = ab
    
    return B, A


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
    lap = np.fix(npt / 25).astype(int)
    mf = F.size
    npt = npt + 1  # For [dc 1 2 ... nyquist].
    Ht = np.array(np.zeros((1, npt)))
    nint = mf - 1
    df = np.diff(F)

    nb = 0
    Ht[0][0] = M[0]
    for i in range(nint):
        if df[i] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.fix(F[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (j - nb) / (ne - nb)

        Ht[0][nb:ne + 1] = np.array(inc * M[i + 1] + (1 - inc) * M[i])
        nb = ne + 1

    Ht = np.concatenate((Ht, Ht[0][-2:0:-1]), axis=None)
    n = Ht.size
    n2 = np.fix((n + 1) / 2)
    nb = order
    nr = 4 * order
    nt = np.arange(0, nr)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[0:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))   # pick NR correlations  # noqa

    # Form window to be used in extracting the right "wing" of two-sided
    # covariance sequence
    Rwindow = np.concatenate(
        (1 / 2, np.ones((1, int(n2 - 1))), np.zeros((1, int(n - n2)))),
        axis=None)
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


def yulewalk_filter(X, B, A, order=None, zi=None, axis=-1):
    # apply the signal shaping filter and initialize the IIR filter state
    if zi is None:
        # zi = signal.lfilter_zi(B, A)
        # zi = np.transpose(X[:, 0] * zi[:, None])
        # Init zero state to mimic matlab, 
        if order is None:
            zi = np.zeros((X.shape[0], X.shape[0] + 1))
        else:
            zi = np.zeros((X.shape[0], order))
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)
    else:
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)

    return out, zf


def fit_eeg_distribution(X, min_clean_fraction=0.25, max_dropout_fraction=0.1, quants=[0.022, 0.6], step_sizes=[0.01, 0.01], beta_range=np.linspace(1.7, 3.5, 13)):

    X = X.copy()
    # sort data so we can access quantiles directly
    X = np.sort(X.ravel())
    n = len(X)
    
    # calculate z bounds and pdf rescaler for the truncated standard generalized Gaussian pdf
    zbounds = {}
    rescale = []
    for b in beta_range:
        zbounds[b] = np.sign(np.array(quants) - 0.5) * gammaincinv(1/b, np.sign(np.array(quants) - 0.5) * (2 * np.array(quants) - 1)) ** (1/b)

        rescale.append(b/(2*gamma(1/b)))
        
    # determine the quantile-dependent limits for the grid search
    lower_min = min(quants)
    max_width = np.diff(quants)[0]
    min_width = min_clean_fraction * max_width
    
    # get matrix of shifted data ranges
    # Compute the indices    
    start_idx = np.round(np.arange(1, int(np.round(n*max_width)) + 1)).astype(int)
    end_idx = np.round(n * np.arange(lower_min, lower_min + max_dropout_fraction + step_sizes[0]*0.5, step_sizes[0])).astype(int)

    # Use the computed indices to slice the array
    X_range = np.array([X[idx + end_idx - 1] for idx in start_idx])

    X1 = X_range[0, :]
    X_range = X_range - X1
    
    opt_val = np.inf
    # for each interval width...
    ms = np.round(n * np.arange(max_width, min_width, -step_sizes[1])).astype(int)
    for m in ms:
        # scale and bin the data in the intervals
        nbins = int(np.round(3 * np.log2(1 + m/2)))
        H = X_range[:m] * (nbins / X_range[m - 1])
        
        # Define bins for the histogram
        bins = np.arange(nbins+1)
        hist_counts = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], 0, H)
        inf_bin_counts = np.apply_along_axis(lambda x: np.sum(x >= (nbins-1)), 0, H)
        hist_counts = np.vstack([hist_counts, inf_bin_counts])
        logq = np.log(hist_counts + 0.01)
     
        # for each shape value...
        for b in beta_range:            
            bounds = zbounds[b]
            
            # evaluate truncated generalized Gaussian pdf at bin centers
            x = bounds[0] + (0.5 + np.arange(0, nbins)) / nbins * np.diff(bounds)
            p = np.exp(-np.abs(x)**b) * rescale[beta_range.tolist().index(b)]
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
                opt_lu = [X1[idx], X1[idx] + X_range[m-1, idx]]
 
    # recover distribution parameters at optimum
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)
    # print(np.array(opt_lu).shape)
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta
    
    # calculate the distribution's standard deviation from alpha and beta
    sig = np.sqrt((alpha**2) * gamma(3/beta) / gamma(1/beta))

    return mu, sig, alpha, beta


def fit_imu_distribution(X, min_clean_fraction=0.3, max_dropout_fraction=0.3, quants=[0.022, 0.6], step_sizes=[0.01, 0.01], beta_range=np.linspace(1.7, 3.5, 13)):
    X = X.copy()
    X = np.sort(X.ravel())
    n = len(X)
    
    # Calculate z bounds and pdf rescaler for the truncated generalized Gaussian pdf
    zbounds = {}
    rescale = []
    for b in beta_range:
        zbounds[b] = np.sign(np.array(quants) - 0.5) * gammaincinv(1/b, np.sign(np.array(quants) - 0.5) * (2 * np.array(quants) - 1)) ** (1/b)
        rescale.append(b/(2*gamma(1/b)))
        
    lower_min = min(quants)
    max_width = np.diff(quants)[0]
    min_width = min_clean_fraction * max_width
    
    start_idx = np.round(np.arange(1, int(np.round(n*max_width)) + 1)).astype(int)
    end_idx = np.round(n * np.arange(lower_min, lower_min + max_dropout_fraction + step_sizes[0]*0.5, step_sizes[0])).astype(int)
    
    X_range = np.array([X[idx + end_idx - 1] for idx in start_idx])
    X1 = X_range[0, :]
    X_range = X_range - X1
    
    opt_val = np.inf
    ms = np.round(n * np.arange(max_width, min_width, -step_sizes[1])).astype(int)
    for m in ms:
        nbins = int(np.round(3 * np.log2(1 + m/2)))
        H = X_range[:m] * (nbins / X_range[m - 1])
        
        bins = np.arange(nbins+1)
        hist_counts = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], 0, H)
        inf_bin_counts = np.apply_along_axis(lambda x: np.sum(x >= (nbins-1)), 0, H)
        hist_counts = np.vstack([hist_counts, inf_bin_counts])
        logq = np.log(hist_counts + 0.01)
        
        for b in beta_range:
            bounds = zbounds[b]
            x = bounds[0] + (0.5 + np.arange(0, nbins)) / nbins * np.diff(bounds)
            p = np.exp(-np.abs(x)**b) * rescale[beta_range.tolist().index(b)]
            p = p / np.sum(p)
            kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m)
            
            min_val = np.min(kl)
            idx = np.argmin(kl)
            
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = b
                opt_bounds = bounds   
                opt_lu = [X1[idx], X1[idx] + X_range[m-1, idx]]
 
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta
    
    sig = np.sqrt((alpha**2) * gamma(3/beta) / gamma(1/beta))

    return mu, sig, alpha, beta


def moving_average(N, X, Zi=None):
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


def denf(R, na):
    """Compute denominator from covariances.

    A = DENF(R,NA) computes order NA denominator A from covariances
    R(0)...R(nr) using the Modified Yule-Walker method. This function is used
    by YULEWALK.

    """
    nr = np.max(np.size(R))
    Rm = toeplitz(R[na:nr - 1], R[na:0:-1])
    Rhs = - R[na + 1:nr]
    A = np.concatenate(
        (1, linalg.lstsq(Rm, Rhs.T, rcond=None)[0].T), axis=None)
    return A