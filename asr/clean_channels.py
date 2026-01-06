import time
import numpy as np
from scipy.signal import firwin, filtfilt
from .asr_utils import filtfilt_fast, design_fir,calc_projector, pop_select, mad


def clean_channels(signal, corr_threshold=0.8, noise_threshold=4, window_len=5, max_broken_time=0.4, 
                   num_samples=50, subset_size=0.25, reset_rng=True):
    
    # Check if 'data' and 'srate' are present in the signal dictionary
    if 'data' not in signal or 'srate' not in signal:
        raise ValueError("The input signal must be a dictionary containing 'data' and 'srate' keys.")
    
    subset_size = round(subset_size * signal['data'].shape[0])
    
    if max_broken_time > 0 and max_broken_time < 1:
        max_broken_time = signal['data'].shape[1] * max_broken_time
    else:
        max_broken_time = round(signal['srate']) * max_broken_time
    
    signal['data'] = signal['data'].astype(np.float64)
    C, S = signal['data'].shape
    window_len = window_len * np.round(signal['srate'])
    wnd = np.arange(window_len)
    
    offsets = np.arange(0, S-window_len, window_len)
    W = len(offsets)
    
    print('Scanning for bad channels...')
    
    if signal['srate'] > 100:
        # remove signal content above 50Hz
        # B = firwin(101, [0, 45, 50], pass_zero='bandpass', fs=signal['srate'])
        # B = firwin(101, [0, 45], pass_zero='bandpass', fs=signal['srate'])

        # Apply the FIR filter to the EEG data
        # X = np.zeros_like(signal['data'].T)
        # for c in range(signal['nbchan']-1, -1, -1):
        #     X[:, c] = filtfilt(B, 1, signal['data'][c, :])
        
        # remove signal content above 50Hz
        B = design_fir(100, [2 * f / signal['srate'] for f in [0, 45, 50, signal['srate']]], [1, 1, 0, 0])
        
        X = np.zeros_like(signal['data'].T)
        
        # Loop over channels
        for c in range(signal['nbchan']-1, -1, -1):            
            X[:, c] = filtfilt_fast(B, 1, signal['data'][c, :])
        
        # determine z-scored level of EM noise-to-signal ratio for each channel
        noisiness = mad(signal['data'].T - X) / mad(X)
        znoise = (noisiness - np.median(noisiness)) / (mad(noisiness) * 1.4826)
        
        # Trim channels based on that
        noise_mask = znoise > noise_threshold
    else:
        X = signal['data'].T
        noise_mask = np.zeros((C, 1), dtype=bool).T  # Create a 'false' array with the same shape
    


    # Step 1: Check if 'X', 'Y', 'Z' fields are present in signal['chanlocs']
    if not all('X' in ch and 'Y' in ch and 'Z' in ch for ch in signal['chanlocs']):
        raise ValueError("To use this function most of your channels should have X, Y, Z location measurements.")

    # Step 2: Extract channel locations and find usable channels
    x = [ch['X'] for ch in signal['chanlocs']]
    y = [ch['Y'] for ch in signal['chanlocs']]
    z = [ch['Z'] for ch in signal['chanlocs']]
    usable_channels = [i for i, (xi, yi, zi) in enumerate(zip(x, y, z)) if xi is not None and yi is not None and zi is not None]

    # Check that more than 50% of the channels are usable
    if len(usable_channels) <= 0.5 * len(signal['chanlocs']):
        raise ValueError("To use this function most of your channels should have X, Y, Z location measurements.")

    # Step 3: Create a 3xN matrix (numpy array) for usable channel locations
    locs = np.array([[x[i] for i in usable_channels],
                    [y[i] for i in usable_channels],
                    [z[i] for i in usable_channels]])

    # Step 4: Index into X using usable_channels
    X = X[:, usable_channels]


    # caculate all-channel reconstruction matrices from random channel subsets   
    if reset_rng:
        np.random.seed(None)
    # Calculate all-channel reconstruction matrices from random channel subsets
    P = calc_projector(locs, num_samples, subset_size)
    
    # Initialize 'corrs' as a zeros array with shape (length of usable_channels, W)
    corrs = np.zeros((len(usable_channels), W))
    
    # calculate each channel's correlation to its RANSAC reconstruction for each window
    time_passed_list = np.zeros(W)
    
    # Loop over the windows
    for o in range(W):
        start_time = time.time()  # Start the timer

        # Extract the window of data
        XX = X[offsets[o] + wnd, :]
        YY = np.sort(np.dot(XX, P).reshape(len(wnd), len(usable_channels), num_samples), axis=2)
        YY = YY[:, :, round(YY.shape[2] / 2)]
        
        # Calculate each channel's correlation to its RANSAC reconstruction for each window
        corrs[:, o] = np.sum(XX * YY, axis=0) / (np.sqrt(np.sum(XX ** 2, axis=0)) * np.sqrt(np.sum(YY ** 2, axis=0)))
        
        # Record the elapsed time for this iteration
        time_passed_list[o] = time.time() - start_time
        median_time_passed = np.median(time_passed_list[:o + 1])
        
        # Print progress and estimated time remaining
        print(f'clean_channel: {o + 1:3.0f}/{W} blocks, {median_time_passed * (W - o) / 60:.1f} minutes remaining.')
        
    # Identify the flagged (bad) channels
    flagged = corrs < corr_threshold
    # Initialize array to mark channels for removal
    removed_channels = np.zeros(C, dtype=bool)
    # Mark all channels for removal which have more flagged samples than the maximum number of ignored samples
    removed_channels[usable_channels] = np.sum(flagged, axis=1) * window_len > max_broken_time
    # Combine with previously identified noisy channels
    removed_channels = removed_channels | noise_mask.T 
        
        
    # Apply removal of bad channels
    if np.mean(removed_channels) > 0.75:
        raise ValueError('More than 75% of your channels were removed -- '
                         'this is probably caused by incorrect channel location measurements '
                         '(e.g., wrong cap design).')
    elif any(removed_channels):
        try:
            signal = pop_select(signal, 'nochannel', np.where(removed_channels)[0])
        except Exception as e:
            # Check if pop_select function exists
            if 'pop_select' not in globals() and 'pop_select' not in locals():
                print("Apparently you do not have EEGLAB's pop_select() available.")
            else:
                print('Could not select channels using EEGLAB\'s pop_select(); details: ')
                print(e)
            
            print(f'Removing {np.sum(removed_channels)} channels and dropping signal meta-data.')
            if len(signal['chanlocs']) == signal['data'].shape[0]:
                signal['chanlocs'] = [ch for ch, rm in zip(signal['chanlocs'], removed_channels) if not rm]
            
            signal['data'] = signal['data'][~removed_channels, :]
            signal['nbchan'] = signal['data'].shape[0]
            
            # Clear associated fields
            for field in ['icawinv', 'icasphere', 'icaweights', 'icaact', 'stats', 'specdata', 'specicaact']:
                signal[field] = None
        
        # Update the clean_channel_mask in the 'etc' field of the signal
        if 'clean_channel_mask' in signal['etc']:
            signal['etc']['clean_channel_mask'][signal['etc']['clean_channel_mask']] = ~removed_channels
        else:
            signal['etc']['clean_channel_mask'] = ~removed_channels 
        
    return signal, removed_channels

