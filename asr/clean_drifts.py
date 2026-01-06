import numpy as np
from scipy.signal import filtfilt, butter, kaiserord, firwin


def clean_drifts_iir(input_sig, transition=[0.5, 1], order=4):
    """
    Removes drifts from the data using a forward-backward high-pass filter.
    
    Parameters:
        input_signal (dict): The continuous data to filter. 
                             It is assumed that input_signal['data'] contains the actual data and 
                             input_signal['srate'] contains the sampling rate.
        transition (list): The transition band in Hz, i.e. lower and upper edge of the transition. 
                           Default is [0.5, 1].
        order (int): The order of the filter. Default is 4.
        
    Returns:
        input_signal (dict): The filtered signal.
    """
    
    input_signal = input_sig.copy()
    # Ensure data is in double precision
    input_signal['data'] = np.array(input_signal['data'], dtype=np.float64)

    # Design highpass IIR filter
    nyquist = 0.5 * input_signal['srate']
    low = transition[0] / nyquist
    high = transition[1] / nyquist
    B, A = butter(order, low, btype='high')

    # Apply the filter, channel by channel to save memory
    input_signal['data'] = filtfilt(B, A, input_signal['data'], padlen=3000)
    
    input_signal['data'] = input_signal['data'].T
    
    # Save the filter kernel in the 'etc' field
    input_signal['etc'] = {'clean_drifts_kernel': B}
        
    return input_signal


def clean_drifts_fir(input_signal, transition=[0.5, 1], attenuation=80):
    """
    Removes drifts from the data using a forward-backward high-pass filter.
    
    Parameters:
        input_signal (dict): The continuous data to filter. 
                             It is assumed that input_signal['data'] contains the actual data and 
                             input_signal['srate'] contains the sampling rate.
        transition (list): The transition band in Hz, i.e. lower and upper edge of the transition. 
                           Default is [0.5, 1].
        attenuation (float): Stop-band attenuation, in db. Default is 80.
        
    Returns:
        input_signal (dict): The filtered signal.
    """
    # Ensure data is in double precision
    input_signal['data'] = np.array(input_signal['data'], dtype=np.float64)

    # Design highpass FIR filter
    nyquist = 0.5 * input_signal['srate']
    transition = [t / nyquist for t in transition]
    
    numtaps, beta = kaiserord(attenuation, transition[1] - transition[0])
    B = firwin(numtaps + 1, transition[1], window=('kaiser', beta), pass_zero='highpass')
    
    # Apply the filter, channel by channel to save memory
    input_signal['data'] = filtfilt(B, 1, input_signal['data'], padlen=999)
    
    input_signal['data'] = input_signal['data'].T
    
    # Save the filter kernel in the 'etc' field
    input_signal['etc'] = {'clean_drifts_kernel': B}
        
    return input_signal