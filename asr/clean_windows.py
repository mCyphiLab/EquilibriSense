import numpy as np
from .asr_utils import fit_eeg_distribution, pop_select
import numba as nb

def clean_windows(signa, max_bad_channels=0.2, zthresholds=[-3.5, 5], window_len=1, window_overlap=0.66,
                  max_dropout_fraction=0.1, min_clean_fraction=0.25, truncate_quant=[0.022, 0.6], 
                  step_sizes=[0.01, 0.01], shape_range=np.linspace(1.7, 3.5, 13)):
    
    signal = signa.copy()
    
    if max_bad_channels > 0 and max_bad_channels < 1:
        max_bad_channels = round(signal['data'].shape[0] * max_bad_channels)
        
    C, S = signal['data'].shape
    
    N = window_len * signal['srate']
    wnd = np.arange(0, N)
    offsets = np.round(np.arange(0, S - N, (N * (1 - window_overlap)))).astype(int)

    # print('Determining time window rejection thresholds...')
    
    wz = np.zeros((C, len(offsets)))
    for c in range(C-1, -1, -1):
        X = signal['data'][c, :] ** 2
        X = np.sqrt(np.sum(X[offsets[:, None] + wnd], axis=1) / N)
        
        mu, sig, _, _ = fit_eeg_distribution(X, min_clean_fraction, max_dropout_fraction, truncate_quant, step_sizes, shape_range)
        
        wz[c, :] = (X - mu) / sig
        
    # print('done.')
    
    swz = np.sort(wz, axis=0)    
    remove_mask = np.zeros(swz.shape[1], dtype=bool)
    
    if max(zthresholds) > 0:
        remove_mask = swz[-max_bad_channels - 1 , :] > max(zthresholds)
    if min(zthresholds) < 0:
        remove_mask |= swz[max_bad_channels, :] < min(zthresholds)
    

    removed_windows = np.where(remove_mask)[0]
    removed_samples = offsets[removed_windows][:, None] + wnd[None, :]
    sample_mask = np.ones(S, dtype=bool)
    
    sample_mask[removed_samples] = False
    
    # print(f"Keeping {100 * np.mean(sample_mask):.1f}% ({np.sum(sample_mask) / signal['srate']:.0f} seconds) of the data.")
    
    int_sample_mask = np.array(sample_mask).astype(int)
    diff_result = np.diff([0] + list(int_sample_mask) + [0])
    starts = np.where(diff_result == 1)[0]
    ends = np.where(diff_result == -1)[0] - 1

    retain_data_intervals = np.column_stack((starts, ends))
    
    # Update the signal dictionary based on the selection
    # try:
    # Filter the remained window
    to_keep = []
    for start, end in retain_data_intervals:
        to_keep.extend(range(start, end+1))
    cleaned_data = signal['data'][:, to_keep]
    signal['data'] = cleaned_data
    # except Exception as e:
        # print("Could not select time windows; details: ", str(e))
        # signal['data'] = signal['data'][:, sample_mask]
        # signal['pnts'] = signal['data'].shape[1]
        # signal['xmax'] = signal['xmin'] + (signal['pnts'] - 1) / signal['srate']
        # signal['event'], signal['urevent'], signal['epoch'], signal['icaact'], signal['reject'], signal['stats'], signal['specdata'], signal['specicaact'] = [], [], [], [], [], [], [], []

    if 'clean_sample_mask' in signal['etc']:
        oneInds = np.where(signal['etc']['clean_sample_mask'] == 1)[0]
        if len(oneInds) == len(sample_mask):
            signal['etc']['clean_sample_mask'][oneInds] = sample_mask
        else:
            print('Warning: EEG.etc.clean_sample is present. It is overwritten.')
    else:
        signal['etc']['clean_sample_mask'] = sample_mask
        
    return signal, sample_mask


from numba import jit, cuda, guvectorize
from .asr_utils_realtime import fit_eeg_distribution_realtime

@jit(target_backend='cuda', nopython=True, cache=True)  
def sort_2d_array(x, axis=0):
    n, m = np.shape(x)
    if axis == 0:
        for col in range(m):
            x[:, col] = np.sort(x[:, col])
    elif axis == 1:
        for row in range(n):
            x[row] = np.sort(x[row])
    else:
        raise ValueError("Invalid axis. Axis must be 0 or 1.")
    
    return x


@jit(target_backend='cuda', nopython=True, cache=True)  
def clean_windows_realtime(data, srate, max_bad_channels=0.2, zthresholds=np.array([-3.5, 5]), window_len=1, window_overlap=0.66,
                  max_dropout_fraction=0.1, min_clean_fraction=0.25, truncate_quant=np.array([0.022, 0.6]), 
                  step_sizes=np.array([0.01, 0.01]), shape_range=np.linspace(1.7, 3.5, 13)):
    
    signal = data.copy()
    if max_bad_channels > 0 and max_bad_channels < 1:
        max_bad_channels = np.round(signal.shape[0] * max_bad_channels)
        max_bad_channels = int(max_bad_channels)
        
    C, S = signal.shape
    
    N = window_len * srate
    wnd = np.arange(0, N)
    offsets = np.round(np.arange(0, S - N, (N * (1 - window_overlap)), dtype=np.int32))

    # print('Determining time window rejection thresholds...')
    
    wz = np.zeros((C, len(offsets)))
    for c in range(C-1, -1, -1):
        X = signal[c, :] ** 2
        X = np.sqrt(np.sum(X[offsets + wnd], axis=1) / N)
        
        mu, sig, _, _ = fit_eeg_distribution_realtime(X, min_clean_fraction, max_dropout_fraction, truncate_quant, step_sizes, shape_range)
        
        wz[c, :] = (X - mu) / sig
        
    # print('done.')
    
    swz = sort_2d_array(wz, axis=0)    
    remove_mask = np.zeros(swz.shape[1], dtype=np.bool_)
    
    if max(zthresholds) > 0:
        remove_mask = swz[int(-max_bad_channels - 1) , :] > max(zthresholds)
    if min(zthresholds) < 0:
        remove_mask |= swz[int(max_bad_channels), :] < min(zthresholds)
    

    removed_windows = np.where(remove_mask)[0]
    removed_samples = offsets[removed_windows][:, None] + wnd[None, :]
    sample_mask = np.ones(S, dtype=np.bool_)
    removed_samples = (offsets[removed_windows][:, None] + wnd[None, :]).ravel()

    sample_mask[removed_samples] = False
    
    # print(f"Keeping {100 * np.mean(sample_mask):.1f}% ({np.sum(sample_mask) / signal['srate']:.0f} seconds) of the data.")

    # Append False to the beginning and end of the array
    padded_sample_mask = np.concatenate((np.array([False]), sample_mask, np.array([False])))

    # Now compute the diff
    diff_result = np.diff(padded_sample_mask)

    starts = np.where(diff_result == 1)[0]
    ends = np.where(diff_result == -1)[0] - 1

    retain_data_intervals = np.column_stack((starts, ends))
    
    # Update the signal dictionary based on the selection
    # try:
    # Filter the remained window
    to_keep = []
    for start, end in retain_data_intervals:
        to_keep.extend(range(start, end+1))
    to_keep = np.array(to_keep)

    cleaned_data = signal[:, to_keep]
    signal = cleaned_data
    # except Exception as e:
        # print("Could not select time windows; details: ", str(e))
        # signal['data'] = signal['data'][:, sample_mask]
        # signal['pnts'] = signal['data'].shape[1]
        # signal['xmax'] = signal['xmin'] + (signal['pnts'] - 1) / signal['srate']
        # signal['event'], signal['urevent'], signal['epoch'], signal['icaact'], signal['reject'], signal['stats'], signal['specdata'], signal['specicaact'] = [], [], [], [], [], [], [], []

        
    return signal, sample_mask
