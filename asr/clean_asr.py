import numpy as np
from .asr_calibrate import asr_calibrate, asr_calibrate_r, asr_calibrate_realtime, asr_calibrate_noise
from .asr_process import asr_process, asr_process_r, asr_process_realtime, asr_process_noise
from .clean_windows import clean_windows, clean_windows_realtime
import time
from numba import jit, cuda


def clean_asr(signal, noise, cutoff=5, windowlen=None, stepsize=None, maxdims=0.66, 
              ref_maxbadchannels=0.075, ref_tolerances=[-3.5, 5.5], ref_wndlen=1, 
              usegpu=False, useriemannian=False, maxmem=64):
    
    signal = signal.copy()
    
    if windowlen is None:
        windowlen = max(0.5, 1.5 * signal['nbchan'] / signal['srate'])
    
    # windowlen = 0.5
    # stepsize = 32
    # maxdims = 0.5
    
    signal['data'] = np.array(signal['data'], dtype=np.float64)
    
    try:
        if isinstance(ref_maxbadchannels, (int, float)) and isinstance(ref_tolerances, list) and isinstance(ref_wndlen, (int, float)):
            # print('Finding a clean section of the data...')
            start_time = time.time()
            ref_section, ref_sample_mask = clean_windows(signal, ref_maxbadchannels, ref_tolerances, ref_wndlen)
            elapsed_time = (time.time() - start_time) * 1000  # multiply by 1000 to convert to milliseconds
            # print("Execution time: ", elapsed_time)
        
        elif ref_maxbadchannels == 'off' or ref_tolerances == 'off' or ref_wndlen == 'off':
            print('Using the entire data for calibration (reference parameters set to "off").')
            ref_section = signal
        elif isinstance(ref_maxbadchannels, dict) and all(k in ref_maxbadchannels for k in ("data", "srate", "chanlocs")):
            print('Using a user-supplied clean section of data.')
            ref_section = ref_maxbadchannels
        else:
            raise ValueError('Unsupported value for argument ref_maxbadchannels.')
    except Exception as e:
        print('An error occurred while trying to identify a subset of clean calibration data from the recording.')
        print('Error details: ', str(e))
        print('Falling back to using the entire data for calibration.')
        ref_section = signal

    # print('Estimating calibration statistics; this may take a while...')
    if useriemannian:
        # Not yet supports useriemannian
        state = asr_calibrate_r(ref_section['data'], ref_section['srate'], cutoff, maxmem=maxmem)
    else:
        start_time = time.time()
        # state = asr_calibrate(noise, ref_section['srate'], cutoff, maxmem=maxmem)
        state = asr_calibrate_noise(noise, ref_section['srate'], cutoff, maxmem=maxmem)

    if stepsize is None:
        stepsize = np.floor(signal['srate'] * windowlen / 2)
    
    # sig = np.concatenate((signal['data'], 2 * signal['data'][:, -1][:, None] - signal['data'][:, -2::-1][:, :int(np.round(windowlen / 2 * signal['srate']))]), axis=1)

    mirror_len = int(np.ceil(windowlen / 2 * signal['srate']))
    mirror_segment = signal['data'][:, -2::-1][:, :mirror_len]
    mirrored = 2 * signal['data'][:, -1][:, None] - mirror_segment

    # Concatenate
    sig = np.concatenate((signal['data'], mirrored), axis=1)

    if useriemannian:
        # Not yet supports useriemannian
        signal['data'], state = asr_process_r(sig, signal['srate'], state, windowlen, windowlen / 2, stepsize, maxdims, maxmem, usegpu)
    else:
        # signal['data'], state = asr_process(sig, signal['srate'], state, windowlen, windowlen / 2, stepsize, maxdims, maxmem, usegpu)
        signal['data'], state, artifacts = asr_process_noise(sig, signal['srate'], state, windowlen, windowlen / 2, stepsize, maxdims, maxmem, usegpu)

    signal['data'] = signal['data'][:, state['carry'].shape[1]:]
    artifacts = artifacts[:, state['carry'].shape[1]:]

    return signal, artifacts


@jit(target_backend='cuda', nopython=True, cache=True)  
def clean_asr_realtime(data, srate, nbchan, sp_mask, cutoff=5, windowlen=None, stepsize=None, maxdims=0.66, 
              ref_maxbadchannels=0.075, ref_tolerances=np.array([-3.5, 5.5]), ref_wndlen=1, 
              usegpu=False, useriemannian=False, maxmem=64):
        
    if windowlen is None:
        windowlen = max(0.5, 1.5 * nbchan / srate)
    
    signal = data.copy()
    signal = signal.astype(np.float64)
    
    # try:
        # if isinstance(ref_maxbadchannels, (int, float)) and isinstance(ref_wndlen, (int, float)):
            # print('Finding a clean section of the data...')
    ref_section, ref_sample_mask = clean_windows_realtime(signal, srate, ref_maxbadchannels, ref_tolerances, ref_wndlen)
        
        # elif ref_maxbadchannels == 'off' or ref_tolerances == 'off' or ref_wndlen == 'off':
        #     print('Using the entire data for calibration (reference parameters set to "off").')
        #     ref_section = signal
        # elif isinstance(ref_maxbadchannels, dict) and all(k in ref_maxbadchannels for k in ("data", "srate", "chanlocs")):
        #     print('Using a user-supplied clean section of data.')
        #     ref_section = ref_maxbadchannels
        # else:
        #     raise ValueError('Unsupported value for argument ref_maxbadchannels.')
    # except Exception as e:
    #     print('An error occurred while trying to identify a subset of clean calibration data from the recording.')
    #     print('Error details: ', str(e))
    #     print('Falling back to using the entire data for calibration.')
    #     ref_section = signal


    # print('Estimating calibration statistics; this may take a while...')
    M, T, B, A, cov, carry, iirstate, last_R, last_trivial = asr_calibrate_realtime(ref_section, srate, cutoff, maxmem=maxmem)

    if stepsize is None:
        stepsize = np.floor(srate * windowlen / 2)
    
    mirror_len = int(np.ceil(windowlen / 2 * srate))
    mirror_segment = signal[:, -2::-1][:, :mirror_len]
    mirrored = 2 * signal[:, -1][:, None] - mirror_segment
    # Concatenate
    sig = np.concatenate((signal, mirrored), axis=1)

    signal, M, T, B, A, cov, carry, iirstate, last_R, last_trivial = asr_process_realtime(sig, srate,  M, T, B, A, cov, carry, iirstate, last_R, last_trivial, windowlen, windowlen / 2, stepsize, maxdims, maxmem, usegpu)
    
    signal = signal[:, carry.shape[1]:]
        
    return signal
