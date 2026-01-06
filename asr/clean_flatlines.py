import numpy as np

def clean_flatlines(signal, max_flatline_duration=5, max_allowed_jitter=20):
    """
    Remove (near-) flat-lined channels.
    
    Parameters:
        signal (dict): Continuous data set, assumed to be appropriately high-passed. 
                      It is assumed that signal['data'] contains the actual data and 
                      signal['srate'] contains the sampling rate.
        max_flatline_duration (float): Maximum tolerated flatline duration. In seconds. 
                                       Default is 5.
        max_allowed_jitter (float): Maximum tolerated jitter during flatlines. 
                                    As a multiple of epsilon. Default is 20.
        
    Returns:
        signal (dict): Data set with flat channels removed.
    """
    # Calculate machine epsilon for float64
    eps = np.finfo(np.float64).eps
    
    # Flag channels
    removed_channels = np.zeros(signal['data'].shape[0], dtype=bool)
    for c in range(signal['data'].shape[0]):
        zero_intervals = np.diff((abs(np.diff(signal['data'][c, :])) < (max_allowed_jitter * eps)).astype(int))
        zero_intervals = np.column_stack((np.where(zero_intervals == 1)[0], np.where(zero_intervals == -1)[0]))
        if np.any((zero_intervals[:, 1] - zero_intervals[:, 0]) > max_flatline_duration * signal['srate']):
            removed_channels[c] = True
    
    # Remove flat-lined channels
    if np.all(removed_channels):
        print('Warning: all channels have a flat-line portion; not removing anything.')
    elif np.any(removed_channels):
        print('Now removing flat-line channels...')
        signal['data'] = signal['data'][~removed_channels, :]
        
        # Optionally, if we have channel information in the signal dictionary, update it:
        # signal['chanlocs'] = [chan for idx, chan in enumerate(signal['chanlocs']) if not removed_channels[idx]]
        
        # Add clean channel mask to the 'etc' field
        if 'etc' in signal and 'clean_channel_mask' in signal['etc']:
            signal['etc']['clean_channel_mask'] = signal['etc']['clean_channel_mask'] & ~removed_channels
        else:
            signal['etc'] = {'clean_channel_mask': ~removed_channels}
    else:
        print('No channels that needs to be removed')
    
    return signal
