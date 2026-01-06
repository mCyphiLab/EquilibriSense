import numpy as np
from .asr_utils import design_fir, filtfilt_fast, design_kaiser, pop_select

def clean_channels_nolocs(signal, min_corr=0.45, ignored_quantile=0.1, window_len=2, 
                          max_broken_time=0.5, linenoise_aware=True):
    
    signal = signal.copy()
    
    if max_broken_time > 0 and max_broken_time < 1:
        max_broken_time = signal['data'].shape[1] * max_broken_time
    else:
        max_broken_time = signal['srate'] * max_broken_time
    
    signal['data'] = signal['data'].astype(np.float64)
    C, S = signal['data'].shape
    window_len = window_len * signal['srate']
    wnd = np.arange(window_len)
    offsets = np.arange(0, S-window_len, window_len)
    W = len(offsets)
    retained = np.arange(C - int(np.ceil(C * ignored_quantile)))
    
    if linenoise_aware:
        Bwnd = design_kaiser(2*45/signal['srate'], 2*50/signal['srate'], 60, True)
        
        if signal['srate'] <= 110:
            raise ValueError('Sampling rate must be above 110 Hz')
        elif signal['srate'] <= 130:
            B = design_fir(len(Bwnd)-1, [2*f/signal['srate'] for f in [0, 45, 50, 55]], [1, 1, 0, 1, 1], Bwnd)
        else:
            B = design_fir(len(Bwnd)-1, [2*f/signal['srate'] for f in [0, 45, 50, 55, 60, 65]], [1, 1, 0, 1, 0, 1, 1], Bwnd)
        
        X = np.zeros_like(signal['data'].T)
        for c in range(signal['nbchan']-1, -1, -1):
            X[:, c] = filtfilt_fast(B, 1, signal['data'][c, :])
    else:
        X = signal['data'].T
    
    flagged = np.zeros((C, W), dtype=bool)
    for o in range(W):
        sortcc = np.sort(np.abs(np.corrcoef(X[offsets[o] + wnd, :].T))).T
        flagged[:, o] = np.all(sortcc[retained, :] < min_corr, axis=0)
    
    removed_channels = (np.sum(flagged, axis=1) * window_len) > max_broken_time
        
    if all(removed_channels):
        print('Warning: all channels are flagged bad according to the used criterion: not removing anything.')
    elif any(removed_channels):
        print('Now removing bad channels...')
        try:
            signal['data'] = pop_select(signal['data'], None, np.where(removed_channels)[0])
        except Exception as e:
            print("Could not select channels using EEGLAB's pop_select(); details: ")
            print(e)
            print('Falling back to a basic substitute and dropping signal meta-data.')
            if len(signal['chanlocs']) == signal['data'].shape[0]:
                signal['chanlocs'] = [ch for ch, rm in zip(signal['chanlocs'], removed_channels) if not rm]
            
            signal['data'] = signal['data'][~removed_channels, :]
            signal['nbchan'] = signal['data'].shape[0]
            
            # Clear associated fields
            for field in ['icawinv', 'icasphere', 'icaweights', 'icaact', 'stats', 'specdata', 'specicaact']:
                signal[field] = None
            
        if 'clean_channel_mask' in signal['etc']:
            signal['etc']['clean_channel_mask'][signal['etc']['clean_channel_mask']] = ~removed_channels
        else:
            signal['etc']['clean_channel_mask'] = ~removed_channels 
            
    return signal, removed_channels