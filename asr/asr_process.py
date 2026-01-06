import numpy as np
from numpy.linalg import eig
import psutil
import math
from scipy.signal import lfilter, lfilter_zi
from .asr_utils import moving_average, hlp_memfree, yulewalk_filter, design_yulewalk_filter
import scipy
from numba import jit, cuda

def asr_process(data, srate, state, windowlen=0.5, lookahead=None, stepsize=32, maxdims=0.66, maxmem=None, usegpu=False):
    """
    Artifact Subspace Reconstruction processing.
    """
    data = data.copy()
    
    C, S = data.shape
    windowlen = max(windowlen, 1.5 * C / srate)
    
    if lookahead is None:
        lookahead = windowlen / 2

    if maxmem is None:
        if usegpu:
            raise NotImplementedError("GPU support is not implemented in this Python version.")
        else:
            maxmem = hlp_memfree() / (2 ** 21)

    if maxdims < 1:
        maxdims = round(C * maxdims)

    if data.size == 0:
        return data, state
    
    N = math.ceil(windowlen * srate)
    P = math.ceil(lookahead * srate)
    T, M, A, B = state['T'], state['M'], state['A'], state['B']

    # if B is None or A is None:
    #     B, A = design_yulewalk_filter(srate)
    #     state['iir'] = np.zeros((C, C))

    # initialize prior filter state by extrapolating available data into the past (if necessary)
    if 'carry' not in state or state['carry'] is None:
        state['carry'] = np.repeat(2 * data[:, :1], P, axis=1) - data[:, 1 + (P + 1 - np.arange(2, P + 2)) % S]

    data = np.concatenate((state['carry'], data), axis=1)
    data[~np.isfinite(data)] = 0
    
    # split up the total sample range into k chunks that will fit in memory
    if maxmem*1024*1024 - C*C*P*8*3 < 0:
        print('Memory too low, increasing it (rejection block size now depends on available memory so it might not be 100% reproducible)...')
        maxmem = hlp_memfree()/(2^21)
        if (maxmem*1024*1024) - (C*C*P*8*3) < 0:
            print('Not enough memory')
    
           
    # Round up the splits
    splits = math.ceil((C*C*S*8*8 + C*C*8*S/stepsize + C*S*8*2 + S*8*5) / (maxmem*1024*1024 - C*C*P*8*3)) # Mysterious. More memory available, less 'splits'
    #  there is something wrong with the formula above (1.6M splits for
    #  subject 2252A of ds003947; capping at 10000 below)
    splits = min(splits, 10000)
    if splits > 1:
        print('Now cleaning data in blocks',splits) 
    
    for i in range(splits):
        _range = np.arange(1 + math.floor((i-1)*S/splits), min(S, math.floor(i*S/splits)) + 1)
        # Temporary
        _range = np.arange(0, S)
        range_shifted = [x + P for x in _range]

        X, state['iir'] = lfilter(B, A, data[:, range_shifted], axis=1, zi=state['iir'])
                                        
        # compute running mean covariance (assuming a zero-mean signal)
        X_reshaped = X[:, np.newaxis, :]
        Xcov_matrix = X_reshaped * X_reshaped.transpose(1, 0, 2)

        Xcov, state['cov'] = moving_average(N, Xcov_matrix.reshape(C*C, -1), state['cov'])

        # extract the subset of time points at which we intend to update
        update_at = np.arange(stepsize, Xcov.shape[1] + stepsize, stepsize)
        update_at = np.minimum(update_at, Xcov.shape[1]).astype(int)
        
        # if there is no previous R (from the end of the last chunk), we estimate it right at the first sample
        if 'last_R' not in state or state['last_R'] is None:
            update_at = np.insert(update_at, 0, 1)
            state['last_R'] = np.eye(C)
            
        update_at -= 1

        Xcov = Xcov[:, update_at]
        Xcov = np.reshape(Xcov, (C, C, -1), order='F')
                    
        # do the reconstruction in intervals of length stepsize (or shorter if at the end of a chunk)
        last_n = 0

        for j in range(len(update_at)):
            # do a PCA to find potential artifact components
            D, V = np.linalg.eig(Xcov[:, :, j])
        
            order = np.argsort(D)
            D = np.real(D[order])
            V = np.real(V[:, order])                            
            
            # determine which components to keep
            # Note the adjustment for Python's 0-based indexing
            
            # print('D: ', D)
            # print('T: ', T)
            # print('V: ', V)
            # print('C: ', C)

            keep = (D < np.sum((T @ V) ** 2, axis=0)) | (np.arange(C) < (C - maxdims))
            trivial = np.all(keep)
                        
            # update the reconstruction matrix R
            if not trivial:                
                R = np.dot(np.dot(M, np.linalg.pinv(np.dot(V.T, M) * keep[:, None])), V.T)
            else:
                R = np.eye(C)

            # apply the reconstruction to intermediate samples (using raised-cosine blending)
            n = update_at[j]
            
            # print(update_at)                        
            if not trivial or not state['last_trivial']:
                subrange = np.arange(last_n + 1, n + 1)
                blend = (1 - np.cos(np.pi * np.arange(1, n - last_n + 1) / (n - last_n))) / 2
                data[:, subrange] = blend * (R @ data[:, subrange]) + (1 - blend) * (state['last_R'] @ data[:, subrange])
            
            last_n, state['last_R'], state['last_trivial'] = n, R, trivial
        
        if splits > 1:
            print('.', end='')

        # End of the outer loop
        if splits > 1:
            print('\n')
    
        # carry the look-ahead portion of the data over to the state (for successive calls)
        state['carry'] = np.concatenate((state['carry'], data[:, -P:]), axis=1)
        state['carry'] = state['carry'][:, -P:]

        # finalize outputs
        outdata = data[:, :-P]
        
        # The state is saved in the 'state' dictionary, which will be equivalent to 'outstate' in MATLAB
        outstate = state

        # Returning outdata and outstate
        return outdata, outstate


def asr_process_noise(data, srate, state, windowlen=0.5, lookahead=None, stepsize=32, maxdims=0.66, maxmem=None, usegpu=False):
    """
    Artifact Subspace Reconstruction processing.
    """
    data = data.copy()
    
    C, S = data.shape
    windowlen = max(windowlen, 1.5 * C / srate)
    
    if lookahead is None:
        lookahead = windowlen / 2

    if maxmem is None:
        if usegpu:
            raise NotImplementedError("GPU support is not implemented in this Python version.")
        else:
            maxmem = hlp_memfree() / (2 ** 21)

    if maxdims < 1:
        maxdims = round(C * maxdims)

    if data.size == 0:
        return data, state
    
    N = math.ceil(windowlen * srate)
    P = math.ceil(lookahead * srate)
    T, M, A, B = state['T'], state['M'], state['A'], state['B']

    # if B is None or A is None:
    #     B, A = design_yulewalk_filter(srate)
    #     state['iir'] = np.zeros((C, C))

    # initialize prior filter state by extrapolating available data into the past (if necessary)
    if 'carry' not in state or state['carry'] is None:
        state['carry'] = np.repeat(2 * data[:, :1], P, axis=1) - data[:, 1 + (P + 1 - np.arange(2, P + 2)) % S]

    data = np.concatenate((state['carry'], data), axis=1)
    data[~np.isfinite(data)] = 0
    
    # split up the total sample range into k chunks that will fit in memory
    if maxmem*1024*1024 - C*C*P*8*3 < 0:
        print('Memory too low, increasing it (rejection block size now depends on available memory so it might not be 100% reproducible)...')
        maxmem = hlp_memfree()/(2^21)
        if (maxmem*1024*1024) - (C*C*P*8*3) < 0:
            print('Not enough memory')
    
           
    # Round up the splits
    splits = math.ceil((C*C*S*8*8 + C*C*8*S/stepsize + C*S*8*2 + S*8*5) / (maxmem*1024*1024 - C*C*P*8*3)) # Mysterious. More memory available, less 'splits'
    #  there is something wrong with the formula above (1.6M splits for
    #  subject 2252A of ds003947; capping at 10000 below)
    splits = min(splits, 10000)
    if splits > 1:
        print('Now cleaning data in blocks',splits) 

    artifacts = np.zeros(data.shape)  # Array to save the artifacts

    for i in range(splits):
        _range = np.arange(1 + math.floor((i-1)*S/splits), min(S, math.floor(i*S/splits)) + 1)
        # Temporary
        _range = np.arange(0, S)
        range_shifted = [x + P for x in _range]

        # X, state['iir'] = lfilter(B, A, data[:, range_shifted], axis=1, zi=state['iir'])
        # X = lfilter(B, A, data[:, range_shifted], axis=1)
        X = data[:, range_shifted]
                                        
        # compute running mean covariance (assuming a zero-mean signal)
        X_reshaped = X[:, np.newaxis, :]
        Xcov_matrix = X_reshaped * X_reshaped.transpose(1, 0, 2)

        Xcov, state['cov'] = moving_average(N, Xcov_matrix.reshape(C*C, -1), state['cov'])

        # extract the subset of time points at which we intend to update
        update_at = np.arange(stepsize, Xcov.shape[1] + stepsize, stepsize)
        update_at = np.minimum(update_at, Xcov.shape[1]).astype(int)
        
        # if there is no previous R (from the end of the last chunk), we estimate it right at the first sample
        if 'last_R' not in state or state['last_R'] is None:
            update_at = np.insert(update_at, 0, 1)
            state['last_R'] = np.eye(C)
            
        update_at -= 1

        Xcov = Xcov[:, update_at]
        Xcov = np.reshape(Xcov, (C, C, -1), order='F')
                    
        # do the reconstruction in intervals of length stepsize (or shorter if at the end of a chunk)
        last_n = 0

        for j in range(len(update_at)):
            # do a PCA to find potential artifact components
            D, V = np.linalg.eig(Xcov[:, :, j])
        
            order = np.argsort(D)
            D = np.real(D[order])
            V = np.real(V[:, order])                            

            keep = (D < np.sum((T @ V) ** 2, axis=0)) | (np.arange(C) < (C - maxdims))
            trivial = np.all(keep)
                        
            # update the reconstruction matrix R
            if not trivial:                
                R = np.dot(np.dot(M, np.linalg.pinv(np.dot(V.T, M) * keep[:, None])), V.T)
            else:
                R = np.eye(C)
                
            # apply the reconstruction to intermediate samples (using raised-cosine blending)
            n = update_at[j]
            
            # print(update_at)    
            if not trivial or not state['last_trivial']:
                subrange = np.arange(last_n + 1, n + 1)
                
                blend = (1 - np.cos(np.pi * np.arange(1, n - last_n + 1) / (n - last_n))) / 2
                
                # artifacts[:, subrange] = (np.eye(C) - R) @ data[:, subrange]
                artifacts[:, subrange] = blend * ((np.eye(C) - R) @ data[:, subrange]) + (1 - blend) * ((np.eye(C) - state['last_R']) @ data[:, subrange])
                
                # original_data = data[:, subrange]
                # # This is actually the noise that the ASR separate the original signal
                data[:, subrange] = blend * (R @ data[:, subrange]) + (1 - blend) * (state['last_R'] @ data[:, subrange])
                # artifacts[:, subrange] = original_data - data[:, subrange]
            
            
            last_n, state['last_R'], state['last_trivial'] = n, R, trivial
        
        if splits > 1:
            print('.', end='')

        # End of the outer loop
        if splits > 1:
            print('\n')
    
        # carry the look-ahead portion of the data over to the state (for successive calls)
        state['carry'] = np.concatenate((state['carry'], data[:, -P:]), axis=1)
        state['carry'] = state['carry'][:, -P:]

        # finalize outputs
        outdata = data[:, :-P]
        artifacts = artifacts[:, :-P]
        
        # The state is saved in the 'state' dictionary, which will be equivalent to 'outstate' in MATLAB
        outstate = state

        # Returning outdata and outstate
        return outdata, outstate, artifacts



from .asr_utils_realtime import hlp_memfree_realtime

@jit(target_backend='cuda', nopython=True, cache=True)  
def asr_process_realtime(data, srate, m, t, b, a, cov, carry, iirstate, last_R, last_trivial, windowlen=0.5, lookahead=None, stepsize=32, maxdims=0.66, maxmem=None, usegpu=False):
    """
    Artifact Subspace Reconstruction processing.
    """
    data = data.copy()
    
    C, S = data.shape
    windowlen = max(windowlen, 1.5 * C / srate)
    
    if lookahead is None:
        lookahead = windowlen / 2

    if maxmem is None:
        if usegpu:
            raise NotImplementedError("GPU support is not implemented in this Python version.")
        else:
            maxmem = hlp_memfree() / (2 ** 21)

    if maxdims < 1:
        maxdims = round(C * maxdims)

    if data.size == 0:
        return data
    
    N = math.ceil(windowlen * srate)
    P = math.ceil(lookahead * srate)
    T, M, A, B = t, m, a, b

    # initialize prior filter state by extrapolating available data into the past (if necessary)
    if carry is None:
        carry = np.repeat(2 * data[:, :1], P, axis=1) - data[:, 1 + (P + 1 - np.arange(2, P + 2)) % S]

    data = np.concatenate((carry, data), axis=1)
    data[~np.isfinite(data)] = 0
    
    # split up the total sample range into k chunks that will fit in memory
    if maxmem*1024*1024 - C*C*P*8*3 < 0:
        print('Memory too low, increasing it (rejection block size now depends on available memory so it might not be 100% reproducible)...')
        maxmem = hlp_memfree_realtime()/(2^21)
        if (maxmem*1024*1024) - (C*C*P*8*3) < 0:
            print('Not enough memory')
    
           
    # Round up the splits
    splits = math.ceil((C*C*S*8*8 + C*C*8*S/stepsize + C*S*8*2 + S*8*5) / (maxmem*1024*1024 - C*C*P*8*3)) # Mysterious. More memory available, less 'splits'
    #  there is something wrong with the formula above (1.6M splits for
    #  subject 2252A of ds003947; capping at 10000 below)
    splits = min(splits, 10000)
    if splits > 1:
        print('Now cleaning data in blocks',splits) 


    if np.any(iirstate) == None:
        zi = lfilter_zi(B, A)
        zi = np.transpose(data[:, 0] * zi[:, None])
    
    for i in range(splits):
        _range = np.arange(1 + math.floor((i-1)*S/splits), min(S, math.floor(i*S/splits)) + 1)
        # Temporary
        _range = np.arange(0, S)
        range_shifted = [x + P for x in _range]

        X, iirstate = lfilter(B, A, data[:, range_shifted], axis=1, zi=iirstate)
                                        
        # compute running mean covariance (assuming a zero-mean signal)
        X_reshaped = X[:, np.newaxis, :]
        Xcov_matrix = X_reshaped * X_reshaped.transpose(1, 0, 2)

        Xcov, cov = moving_average(N, Xcov_matrix.reshape(C*C, -1), cov)

        # extract the subset of time points at which we intend to update
        update_at = np.arange(stepsize, Xcov.shape[1] + stepsize, stepsize)
        update_at = np.minimum(update_at, Xcov.shape[1]).astype(int)
        
        # if there is no previous R (from the end of the last chunk), we estimate it right at the first sample
        if last_R is None:
            update_at = np.insert(update_at, 0, 1)
            last_R = np.eye(C)
            
        update_at -= 1

        Xcov = Xcov[:, update_at]
        Xcov = np.reshape(Xcov, (C, C, -1), order='F')
                    
        # do the reconstruction in intervals of length stepsize (or shorter if at the end of a chunk)
        last_n = 0

        for j in range(len(update_at)):
            # do a PCA to find potential artifact components
            D, V = np.linalg.eig(Xcov[:, :, j])
        
            order = np.argsort(D)
            D = np.real(D[order])
            V = np.real(V[:, order])                            
            
            # determine which components to keep
            # Note the adjustment for Python's 0-based indexing
            keep = (D < np.sum((T @ V) ** 2, axis=0)) | (np.arange(C) < (C - maxdims))
            trivial = np.all(keep)
                        
            # update the reconstruction matrix R
            if not trivial:                
                R = np.dot(np.dot(M, np.linalg.pinv(np.dot(V.T, M) * keep[:, None])), V.T)
            else:
                R = np.eye(C)

            # apply the reconstruction to intermediate samples (using raised-cosine blending)
            n = update_at[j]
            
            # print(update_at)                        
            if last_trivial is None:
                subrange = np.arange(last_n + 1, n + 1)
                blend = (1 - np.cos(np.pi * np.arange(1, n - last_n + 1) / (n - last_n))) / 2
                data[:, subrange] = blend * (R @ data[:, subrange]) + (1 - blend) * (last_R @ data[:, subrange])
            
            last_n, last_R, last_trivial = n, R, trivial
        
        if splits > 1:
            print('.', end='')

        # End of the outer loop
        if splits > 1:
            print('\n')
    
        # carry the look-ahead portion of the data over to the state (for successive calls)
        carry = np.concatenate((carry, data[:, -P:]), axis=1)
        carry = carry[:, -P:]

        # finalize outputs
        outdata = data[:, :-P]
        
        # Returning outdata and outstate
        return outdata, m, t, b, a, cov, carry, iirstate, last_R, last_trivial




def asr_process_r(data, srate, state, windowlen=0.1, lookahead=None, stepsize=4, maxdims=1, maxmem=None, usegpu=False):
    """
    Artifact Subspace Reconstruction processing.
    """
    data = data.copy()
    
    C, S = data.shape
    windowlen = max(windowlen, 1.5 * C / srate)
    
    if lookahead is None:
        lookahead = windowlen / 2

    if maxmem is None:
        if usegpu:
            raise NotImplementedError("GPU support is not implemented in this Python version.")
        else:
            maxmem = hlp_memfree() / (2 ** 21)

    if maxdims < 1:
        maxdims = round(C * maxdims)

    if data.size == 0:
        return data, state
    
    N = math.ceil(windowlen * srate)
    P = math.ceil(lookahead * srate)
    T, M, A, B = state['T'], state['M'], state['A'], state['B']

    # initialize prior filter state by extrapolating available data into the past (if necessary)
    if 'carry' not in state or state['carry'] is None:
        state['carry'] = np.repeat(2 * data[:, :1], P, axis=1) - data[:, 1 + (P + 1 - np.arange(2, P + 2)) % S]

    data = np.concatenate((state['carry'], data), axis=1)
    data[~np.isfinite(data)] = 0
    
    # split up the total sample range into k chunks that will fit in memory
    if maxmem*1024*1024 - C*C*P*8*3 < 0:
        print('Memory too low, increasing it (rejection block size now depends on available memory so it might not be 100% reproducible)...')
        maxmem = hlp_memfree()/(2^21)
        if (maxmem*1024*1024) - (C*C*P*8*3) < 0:
            print('Not enough memory')
    
           
    # Round up the splits
    splits = math.ceil((C*C*S*8*8 + C*C*8*S/stepsize + C*S*8*2 + S*8*5) / (maxmem*1024*1024 - C*C*P*8*3)) # Mysterious. More memory available, less 'splits'
    #  there is something wrong with the formula above (1.6M splits for
    #  subject 2252A of ds003947; capping at 10000 below)
    splits = min(splits, 10000)
    if splits > 1:
        print('Now cleaning data in blocks',splits) 


    if np.any(state['iir']) == None:
        zi = lfilter_zi(B, A)
        zi = np.transpose(data[:, 0] * zi[:, None])
    
    for i in range(splits):
        _range = np.arange(1 + math.floor((i-1)*S/splits), min(S, math.floor(i*S/splits)) + 1)
        # Temporary
        _range = np.arange(0, S)
        range_shifted = [x + P for x in _range]

        X, state['iir'] = lfilter(B, A, data[:, range_shifted], axis=1, zi=state['iir'])
        
        Y = X
                                        
        # compute running mean covariance (assuming a zero-mean signal)
        X_reshaped = X[:, np.newaxis, :]
        Xcov_matrix = X_reshaped * X_reshaped.transpose(1, 0, 2)

        Xcov, state['cov'] = moving_average(N, Xcov_matrix.reshape(C*C, -1), state['cov'])

        # extract the subset of time points at which we intend to update
        update_at = np.arange(stepsize, Xcov.shape[1] + stepsize, stepsize)
        update_at = np.minimum(update_at, Xcov.shape[1]).astype(int)
        
        # if there is no previous R (from the end of the last chunk), we estimate it right at the first sample
        if 'last_R' not in state or state['last_R'] is None:
            update_at = np.insert(update_at, 0, 1)
            state['last_R'] = np.eye(C)
            
        update_at -= 1

        Xcov = Xcov[:, update_at]
        Xcov = np.reshape(Xcov, (C, C, -1), order='F')
                    
        # do the reconstruction in intervals of length stepsize (or shorter if at the end of a chunk)
        last_n = 0

        for j in range(len(update_at)):
            # do a PCA to find potential artifact components
            D, V = np.linalg.eig(Xcov[:, :, j])
        
            order = np.argsort(D)
            D = np.real(D[order])
            V = np.real(V[:, order])                            
            
            # determine which components to keep
            # Note the adjustment for Python's 0-based indexing
            keep = (D < np.sum((T @ V) ** 2, axis=0)) | (np.arange(C) < (C - maxdims))
            trivial = np.all(keep)
                        
            # update the reconstruction matrix R
            if not trivial:                
                R = np.dot(np.dot(M, np.linalg.pinv(np.dot(V.T, M) * keep[:, None])), V.T)
            else:
                R = np.eye(C)

            # apply the reconstruction to intermediate samples (using raised-cosine blending)
            n = update_at[j]
            
            # print(update_at)                        
            if not trivial or not state['last_trivial']:
                subrange = np.arange(last_n + 1, n + 1)
                blend = (1 - np.cos(np.pi * np.arange(1, n - last_n + 1) / (n - last_n))) / 2
                data[:, subrange] = blend * (R @ data[:, subrange]) + (1 - blend) * (state['last_R'] @ data[:, subrange])
            
            last_n, state['last_R'], state['last_trivial'] = n, R, trivial
        
        if splits > 1:
            print('.', end='')

        # End of the outer loop
        if splits > 1:
            print('\n')
    
        # carry the look-ahead portion of the data over to the state (for successive calls)
        state['carry'] = np.concatenate((state['carry'], data[:, -P:]), axis=1)
        state['carry'] = state['carry'][:, -P:]

        # finalize outputs
        outdata = data[:, :-P]
        
        # The state is saved in the 'state' dictionary, which will be equivalent to 'outstate' in MATLAB
        outstate = state

        # Returning outdata and outstate
        return outdata, outstate
        # Returning outdata and outstate
        return outdata, outstate, Y