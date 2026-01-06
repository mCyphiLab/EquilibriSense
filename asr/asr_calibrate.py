import numpy as np
from scipy.signal import lfilter, lfilter_zi
from scipy.linalg import sqrtm, eig
from scipy.optimize import minimize
from scipy import signal
import math
from .asr_utils import hlp_memfree, design_yulewalk_filter, block_geometric_median, fit_eeg_distribution, yulewalk_filter, geometric_median, fit_imu_distribution
from .rasr_nonlinear_eigenspace import rasr_nonlinear_eigenspace
import scipy

def asr_calibrate(X, srate, cutoff=5, blocksize=10, B=None, A=None, 
                  window_len=0.5, window_overlap=0.66, max_dropout_fraction=0.1, 
                  min_clean_fraction=0.25, maxmem=None):
    
    C, S = X.shape
        
    if maxmem is None:
        maxmem = hlp_memfree() / (2**21)

    if blocksize is None:
        blocksize = 10
    
    blocksize = max(blocksize, math.ceil((C*C*S*8*3*2)/(maxmem*(2**21))))
        
    X[np.isinf(X) | np.isnan(X)] = 0

    # Apply the signal shaping filter and initialize the IIR filter state
    # try to use yulewalk to design the filter (Signal Processing toolbox required)
    if B is None or A is None:
        B, A = design_yulewalk_filter(srate)
    
    X, z0 = yulewalk_filter(X, B, A, order=8)
    X = X.T
    
    if not np.all(np.isfinite(X)):
        raise ValueError('The IIR filter diverged on your data.')
    
    # Calculate the sample covariance matrices U
    U = np.zeros((len(range(1, S, blocksize)), C * C))

    for k in range(blocksize):
        _range = np.arange(min(S, k), S + k-1, blocksize)
        _range = np.clip(_range, None, S - 1)
        tmp = X[_range, :, None] * X[_range, None, :]
        U += tmp.reshape(U.shape)
    
    # Get the mixing matrix M
    med = block_geometric_median(U/blocksize)
    if np.isnan(med[0]) and U.shape[1] == 1:
        med = np.median(U)
    M = scipy.linalg.sqrtm(np.real(med.reshape((C, C))))
    
    # Window length for calculating thresholds
    N = round(window_len * srate)
    
    if S < N:
        raise ValueError('Not enough reference data.')
    
    # Get the threshold matrix T
    # print('Determining per-component thresholds...')
    D, V = np.linalg.eig(M)

    order = np.argsort(D)
    D = np.real(D[order])
    V = np.real(V[:, order])     

    X = np.abs(X.dot(V))
    
    mu = np.zeros(C)
    sig = np.zeros(C)

    for c in range(C-1, -1, -1):
        # Compute squared amplitude 
        rms = X[:, c] ** 2
                
        # Start indices for each window
        offsets = np.round(np.arange(0, S - N, N * (1 - window_overlap))).astype(int)
        
        # Create a 2D array where each row represents the indices of one window over the rms array
        window_idxs = np.array([offset + np.arange(0, N) for offset in offsets])

        # Compute RMS values for each window
        rms_values = np.sqrt(np.sum(rms[window_idxs], axis=1) / N)

        # Fit a distribution to the clean part
        if len(rms) == 1:
            raise ValueError('Not enough reference data.')

        mu[c], sig[c], _, _ = fit_eeg_distribution(rms_values, min_clean_fraction, max_dropout_fraction)


    # T = np.diag(mu + cutoff*sig).dot(V.T)
    T = np.dot(np.diag(mu + cutoff * sig), V.T)
    # print('done.')
    
    # Initialize the remaining filter state
    state = {
        'M': M, 
        'T': T, 
        'B': B, 
        'A': A, 
        'cov': None, 
        'carry': None, 
        'iir': z0, 
        'last_R': None, 
        'last_trivial': True
    }
    
    print('Results of states retrieved from the calibration')
    print(M.shape)
    print(T.shape)
    print(B.shape)
    print(A.shape)
    
    return state


def asr_calibrate_noise(X, srate, cutoff=5, blocksize=10, B=None, A=None, 
                  window_len=0.5, window_overlap=0.66, max_dropout_fraction=0.1, 
                  min_clean_fraction=0.25, maxmem=None):
    
    C, S = X.shape

    if maxmem is None:
        maxmem = hlp_memfree() / (2**21)

    if blocksize is None:
        blocksize = 10
    
    blocksize = max(blocksize, math.ceil((C*C*S*8*3*2)/(maxmem*(2**21))))
        
    X[np.isinf(X) | np.isnan(X)] = 0

    # Apply the signal shaping filter and initialize the IIR filter state
    # try to use yulewalk to design the filter (Signal Processing toolbox required)
    if B is None or A is None:
        B, A = design_yulewalk_filter(srate)
    
    X, z0 = yulewalk_filter(X, B, A, order=8)
    X = X.T
    
    if not np.all(np.isfinite(X)):
        raise ValueError('The IIR filter diverged on your data.')
    
    # Calculate the sample covariance matrices U
    U = np.zeros((len(range(1, S, blocksize)), C * C))
    for k in range(blocksize):
        _range = np.arange(min(S, k), S + k-1, blocksize)
        _range = np.clip(_range, None, S - 1)
        tmp = X[_range, :, None] * X[_range, None, :]
        U += tmp.reshape(U.shape)
    
    # Get the mixing matrix M
    med = block_geometric_median(U/blocksize)
    if np.isnan(med[0]) and U.shape[1] == 1:
        med = np.median(U)
    
    M = scipy.linalg.sqrtm(np.real(med.reshape((C, C))))
    
    # Window length for calculating thresholds
    N = round(window_len * srate)
    
    if S < N:
        raise ValueError('Not enough reference data.')
    
    # Get the threshold matrix T
    # print('Determining per-component thresholds...')
    D, V = np.linalg.eig(M)

    order = np.argsort(D)
    D = np.real(D[order])
    V = np.real(V[:, order])     

    X = np.abs(X.dot(V))
    
    mu = np.zeros(C)
    sig = np.zeros(C)

    for c in range(C-1, -1, -1):
        # Compute squared amplitude 
        rms = X[:, c] ** 2
                
        # Start indices for each window
        offsets = np.round(np.arange(0, S - N, N * (1 - window_overlap))).astype(int)
        
        # Create a 2D array where each row represents the indices of one window over the rms array
        window_idxs = np.array([offset + np.arange(0, N) for offset in offsets])

        # Compute RMS values for each window
        rms_values = np.sqrt(np.sum(rms[window_idxs], axis=1) / N)

        # Fit a distribution to the clean part
        if len(rms) == 1:
            raise ValueError('Not enough reference data.')

        mu[c], sig[c], _, _ = fit_imu_distribution(rms_values, min_clean_fraction, max_dropout_fraction)


    # T = np.diag(mu + cutoff*sig).dot(V.T)
    T = np.dot(np.diag(mu + cutoff * sig), V.T)
    # print('done.')
    
    # Initialize the remaining filter state
    state = {
        'M': M, 
        'T': T, 
        'B': None,
        'A': None,
        'cov': None, 
        'carry': None, 
        'iir': None, 
        'last_R': None, 
        'last_trivial': True
    }
    
       
    return state



from numba import jit, cuda
from .asr_utils_realtime import fit_eeg_distribution_realtime, yulewalk_filter_realtime, design_yulewalk_filter_realtime, block_geometric_median_realtime, hlp_memfree_realtime
    
@jit(target_backend='cuda', nopython=True, cache=True)  
def asr_calibrate_realtime(X, srate, cutoff=5, blocksize=10, B=None, A=None, 
                  window_len=0.5, window_overlap=0.66, max_dropout_fraction=0.1, 
                  min_clean_fraction=0.25, maxmem=None):
    
    C, S = X.shape
    
    if maxmem is None:
        maxmem = 64

    if blocksize is None:
        blocksize = 10
    
    blocksize = max(blocksize, math.ceil((C*C*S*8*3*2)/(maxmem*(2**21))))
        
    # X[np.isinf(X) | np.isnan(X)] = 0
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isinf(X[i, j]) or np.isnan(X[i, j]):
                X[i, j] = 0

    # Apply the signal shaping filter and initialize the IIR filter state
    #  try to use yulewalk to design the filter (Signal Processing toolbox required)
    if B is None or A is None:
        B, A = design_yulewalk_filter_realtime(srate)
    
    X, z0 = yulewalk_filter_realtime(X, B, A)
    X = X.T
    
    # if not np.all(np.isfinite(X)):
    #     raise ValueError('The IIR filter diverged on your data.')
    
    # Calculate the sample covariance matrices U
    U = np.zeros((len(range(1, S, blocksize)), C * C))
    for k in range(blocksize):
        _range = np.arange(min(S, k), S + k-1, blocksize)
        _range = np.clip(_range, None, S - 1)
        tmp = X[_range, :, None] * X[_range, None, :]
        U += tmp.reshape(U.shape)
    
    # Get the mixing matrix M
    med = block_geometric_median_realtime(U/blocksize)
    if np.isnan(med[0]) and U.shape[1] == 1:
        med = np.median(U)
    M = scipy.linalg.sqrtm(np.real(med.reshape((C, C))))
    
    # Window length for calculating thresholds
    N = round(window_len * srate)
    
    if S < N:
        raise ValueError('Not enough reference data.')
    
    # Get the threshold matrix T
    # print('Determining per-component thresholds...')
    D, V = np.linalg.eig(M)

    order = np.argsort(D)
    D = np.real(D[order])
    V = np.real(V[:, order])     

    X = np.abs(X.dot(V))
    
    mu = np.zeros(C)
    sig = np.zeros(C)

    for c in range(C-1, -1, -1):
        # Compute squared amplitude 
        rms = X[:, c] ** 2
                
        # Start indices for each window
        offsets = np.round(np.arange(0, S - N, N * (1 - window_overlap))).astype(int)
        
        # Create a 2D array where each row represents the indices of one window over the rms array
        window_idxs = np.array([offset + np.arange(0, N) for offset in offsets])

        # Compute RMS values for each window
        rms_values = np.sqrt(np.sum(rms[window_idxs], axis=1) / N)

        # Fit a distribution to the clean part
        if len(rms) == 1:
            raise ValueError('Not enough reference data.')

        mu[c], sig[c], _, _ = fit_eeg_distribution_realtime(rms_values, min_clean_fraction, max_dropout_fraction)


    # T = np.diag(mu + cutoff*sig).dot(V.T)
    T = np.dot(np.diag(mu + cutoff * sig), V.T)
    # print('done.')
        
    return M, T, B, A, None, None, z0, None, True


def block_covariance(data, window=128, overlap=0.5, padding=True,
                     estimator='cov'):
    """Compute blockwise covariance.

    Parameters
    ----------
    data : array, shape=(n_chans, n_samples)
        Input data (must be 2D)
    window : int
        Window size.
    overlap : float
        Overlap between successive windows.

    Returns
    -------
    cov : array, shape=(n_blocks, n_chans, n_chans)
        Block covariance.

    """
    from pyriemann.utils.covariance import _check_est

    assert 0 <= overlap < 1, "overlap must be < 1"
    est = _check_est(estimator)
    cov = []
    n_chans, n_samples = data.shape
    if padding:  # pad data with zeros
        pad = np.zeros((n_chans, int(window / 2)))
        data = np.concatenate((pad, data, pad), axis=1)

    jump = int(window * overlap)
    ix = 0
    while (ix + window < n_samples):
        cov.append(est(data[:, ix:ix + window]))
        ix = ix + jump

    return np.array(cov)




# Not yet supported
def asr_calibrate_r(X, srate, cutoff=3, blocksize=10, B=None, A=None, 
                  window_len=0.1, window_overlap=0.5, max_dropout_fraction=0.1, 
                  min_clean_fraction=0.3, maxmem=None):
    
    C, S = X.shape
    
    if maxmem is None:
        maxmem = hlp_memfree() / (2**21)

    if blocksize is None:
        blocksize = 10
    
    blocksize = max(blocksize, math.ceil((C*C*S*8*3*2)/(maxmem*(2**21))))
        
    X[np.isinf(X) | np.isnan(X)] = 0

    # Apply the signal shaping filter and initialize the IIR filter state
    #  try to use yulewalk to design the filter (Signal Processing toolbox required)
    if B is None or A is None:
        B, A = design_yulewalk_filter(srate)
    
    X, z0 = yulewalk_filter(X, B, A, order=8)
    X = X.T
    
    # if not np.all(np.isfinite(X)):
    #     raise ValueError('The IIR filter diverged on your data.')
    
    # Calculate the sample covariance matrices U
    indx = np.arange(0, S, blocksize)
    U = np.zeros((len(indx), C * C))

    for k, idx in enumerate(indx):
        _range = np.arange(idx, min(S, idx + blocksize))
        tmp = np.matmul(X[_range, :].T, X[_range, :])
        U[k, :] = tmp.ravel()
    
    # Get the mixing matrix M
    med = block_geometric_median(U/blocksize)
    M = scipy.linalg.sqrtm(np.real(med.reshape((C, C))))
    
    # Window length for calculating thresholds
    N = round(window_len * srate)
    
    # Get the threshold matrix T
    print('Determining per-component thresholds...')
    # D, V = np.linalg.eig(M))     
    
    V, D = rasr_nonlinear_eigenspace(M, C)

    order = np.argsort(D)
    D = np.real(D[order])
    V = np.real(V[:, order])

    X = np.abs(X.dot(V))
    
    mu = np.zeros(C)
    sig = np.zeros(C)

    for c in range(C-1, -1, -1):
        # Compute squared amplitude 
        rms = X[:, c] ** 2
                
        # Start indices for each window
        offsets = np.round(np.arange(0, S - N, N * (1 - window_overlap))).astype(int)
        
        # Create a 2D array where each row represents the indices of one window over the rms array
        window_idxs = np.array([offset + np.arange(0, N) for offset in offsets])

        # Compute RMS values for each window
        rms_values = np.sqrt(np.sum(rms[window_idxs], axis=1) / N)

        # Fit a distribution to the clean part
        if len(rms) == 1:
            raise ValueError('Not enough reference data.')

        mu[c], sig[c], _, _ = fit_eeg_distribution(rms_values, min_clean_fraction, max_dropout_fraction)


    T = np.dot(np.diag(mu + cutoff * sig), V.T)
    # print('done.')
    
    # Initialize the remaining filter state
    state = {
        'M': M, 
        'T': T, 
        'B': B, 
        'A': A, 
        'cov': None, 
        'carry': None, 
        'iir': z0, 
        'last_R': None, 
        'last_trivial': True
    }
    
    return state
