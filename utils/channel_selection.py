import numpy as np
import scipy
from scipy.fft import fft, fftfreq
from scipy.integrate import simps
from utils.signal_processing import *


def compute_band_power_information(datum, fs):
    '''
    Get's EEG relevant bandpower information from a seignal segment.
    
    input:
        datum (np.ndarray): segment of biosignal data
        fs (int): sampling rate of data
    output:
        bp (np.ndarray): bandpowers in the 0.5-4, 8-13, 13-35, and 35-100 Hz frequency ranges
        bp_rel (np.ndarray): normalized bandpowers in the 0.5-4, 4-8, 13-35, and 35-100 Hz frequency ranges
    '''
    bands = [[0.5,4],[8,13],[13,35],[35,100]]

    freq_low = 0.5
    freq_high = 100

    fft_datum = np.abs(fft(datum))
    freqs = fftfreq(len(datum),1/fs)
    indice = np.bitwise_and(freqs<=(fs/2.), freqs>=0)
    fft_datum = fft_datum[indice]
    freqs = freqs[indice]
    total_pow = simps(fft_datum,freqs)
    
    if np.allclose(total_pow, 0):
        return [0,0,0,0], [0,0,0,0]

    bp = []
    bp_rel = []
    for idx in range(len(bands)):
        indice = np.bitwise_and(freqs<=bands[idx][1], freqs>=bands[idx][0])
        power = simps(fft_datum[indice],freqs[indice])
        bp.append(power)
        bp_rel.append(power/total_pow)
        
    return bp, bp_rel

def compute_signal_quality_features(signal_segment, fs):
    '''
    Computes features needed for estimating signal quality of a given signal segment.
    
    input:
        signal_segment (np.ndarray): segment of biosignal data
        fs (int): sampling rate of data
    output:
        features (np.ndarray): array of computed features
    '''
    bandpowers, relative_bandpowers = compute_band_power_information(signal_segment, fs)
    delta, alpha, beta, high = bandpowers
    delta_relative, alpha_relative, beta_relative, high_relative = relative_bandpowers
    features = [delta, alpha, beta, high, delta_relative, \
                alpha_relative, beta_relative, high_relative]

    return np.array(features)


def is_signal_noisy(filtered_segment, fs):
    '''
    Applies rules to evaluate whether or not a given signal segment is noisy.  Thresholds and rules were
    determined with statistical analysis and decision tree fitting.
    
    input:
        filtered_segment (np.ndarray): a signal segment that has already been filtered (notch filter + bandpass filter)
        fs (int): sampling rate of signal data
    output:
        A boolean value, True if the signal segment was inferred to be noisy, False otherwise.
    '''
    signal_quality_features = compute_signal_quality_features(filtered_segment, fs)
    
    if np.max(np.abs(filtered_segment)) < 5:
        return True
    if np.max(np.abs(filtered_segment)) >= 350:
        return True
    if signal_quality_features[7] >= 0.2:
        # Normalized 35-100 Hz bandpower >= 0.2
        return True
    
    if signal_quality_features[0] <= 22076.72:
        # Absolute delta bandpower <= 22076.72
        if signal_quality_features[1] <= 6033.41:
            # Absolute alpha bandpower <= 6033.41
            return False
        else:
            return True
    else:
        if signal_quality_features[0] <= 24951.07:
            # Absolute delta bandpower <= 24951.07
            if signal_quality_features[6] <= 0.13:
                # Normalized beta bandpower <= 0.13
                return True
            else:
                return False
        else:
            return True
        


def get_signal_status(filtered_segment, fs, bad_signal_counter=0):
    '''
    Get the signal quality status (OK, Bad Signal (Bad contact), or Bad Signal (No contact)) of an input signal segment.
    
    input:
        filtered_segment (np.ndarray): Filtered segment of biosignal data
        fs (int): sampling rate of biosignal data
        bad_signal_counter (int): A count of the number of bad signal segments in a row immediately before the current one.
    output:
        signal_status (int): 0 if signal is ok, 1 if signal is noisy with bad contact, 
                             2 if signal is noisy with no contact.
        bad_signal_counter (int): An updated count of the number of bad signal segments in a row including the current one.
    '''
    noisy = is_signal_noisy(filtered_segment, fs)
    
    if not noisy:
        bad_signal_counter = 0
        signal_status = 0
    else:
        bad_signal_counter += 1
        signal_status = 1
        # If the signal has been bad for at least 3 counts and the 100uV crossing frequency >0 10Hz
        crossing_rate_100uv = np.sum(np.diff(np.sign(filtered_segment-100)) != 0)
        if (bad_signal_counter >= 3) and (crossing_rate_100uv >= ((len(filtered_segment)/fs)*10)):
            signal_status = 2
    return signal_status, bad_signal_counter


def is_epoch_signal_bad(epoch_signal, fs, bad_segment_threshold=0, return_status=False):
    '''
    Given an 30-second segment of signal data from one channel, determine if the signal data for this epoch is bad.
    
    input:
        epoch_signal (np.ndarray): Epoch signal segment.
        fs (int): sampling rate of signal data.
        bad_segment_threshold (int): Maximum number of 2-second signal segments for which the signal is noisy.
    output:
        noisy (boolean): True if the epoch signal is bad, false otherwise.
    '''
    epoch_signal = notch_filter(epoch_signal, fs, stop_fs=60, Q=12)
    epoch_signal = notch_filter(epoch_signal, fs, stop_fs=50, Q=10)
    epoch_signal = notch_filter(epoch_signal, fs, stop_fs=25, Q=5)
    epoch_signal = butter_bandpass_filter(epoch_signal, 0.5, 50, fs, order=1)
    
    num_bad_segments = 0
    bad_signal_counter = 0
    segment_window_length = int(fs*10)
    segment_window_stride = int(fs*10)

    for i in range(0, len(epoch_signal), segment_window_stride):
        if i+segment_window_length > len(epoch_signal): break
        signal_segment = epoch_signal[i:i+segment_window_length]
        signal_status, bad_signal_counter = get_signal_status(signal_segment, fs, bad_signal_counter)
        if signal_status > 0:
            num_bad_segments += 1
        if num_bad_segments > bad_segment_threshold:
            break
    noisy = False
    if num_bad_segments > bad_segment_threshold:
        noisy = True
        
    if return_status:
        return noisy, signal_status
    else:
        return noisy


# Main function
if __name__ == "__main__":
    dataset = 'Head_Turn_Testing'
    fs = 250  
    
    import glob
    import pandas as pd
    file_names = sorted(glob.glob(f'./data/logging/{dataset}/CONVERTED_*'))
    data = pd.concat([pd.read_csv(file_name, sep='\t') for file_name in file_names])
    # Drop unneeded data
    data = data.drop(columns=['index', 'EDA', 'time stamp', 'red led', 'ir led', 'green', 'temperature', 'EDA', 'EMG (Right neck muscle)', 'EEG - Pz'])
    channel_names = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4']
    signal_status = {'F3':[], 'Fz':[], 'F4':[], 'C3':[], 'Cz':[], 'C4':[]}
    
    signals = []
    # Segement the signal to each 30s epoch before applying the is_epoch_signal_bad
    epoch = 10 # 10s
    epoch_window_length = int(fs*epoch)
    epoch_window_stride = int(fs*epoch)

    # Append the signal to numpy array
    for i, name in enumerate(channel_names):
        signal = data.iloc[:, i].to_numpy().copy()
        signals.append(signal)
        
        # Apply the is_epoch_signal_bad
        for i in range(0, len(signal), epoch_window_stride):
            if i+epoch_window_length > len(signal): break
            signal_epoch = signal[i:i+epoch_window_length]
            status = is_epoch_signal_bad(signal_epoch, fs)
            signal_status[name].append(status)
        
    signals = np.array(signals)
    
    # Visualize the signal here ?
    