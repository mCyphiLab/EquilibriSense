import numpy as np
import scipy
from scipy import signal
from scipy.signal import butter, sosfilt, sosfilt_zi, iirfilter, filtfilt
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations

EPOCH_LENGTH = 30

STAGE_LABEL_MAP = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, \
                   'Sleep stage N3': 3, 'Sleep stage R': 4, 'Sleep stage ?': -1}

STAGE_LABEL_MAP_4_STAGE = {'Sleep stage W': 0, 'Sleep stage Light Sleep': 1, 'Sleep stage Deep Sleep': 2, \
                           'Sleep stage R': 3, 'Sleep stage ?': -1}

notch_filter_fc = 50
notch_filter_q = 10

def notch_filter(data, fs, stop_fs=notch_filter_fc, Q=notch_filter_q):
    b, a = scipy.signal.iirnotch(stop_fs, Q, fs)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter(order, [lowcut, highcut], btype='band',fs=fs)
    y = signal.filtfilt(b, a, data)
    return y

def iir_bandpass_filter(data, fs, frequency_band=[1,20], order=4):
    '''
    IIR Bandpass filter

    Derived from:
    https://github.com/Dreem-Organization/dreem-learning-open/blob/632a84c7e412f69f51c407ecb2ea91403f0d26c3/dreem_learning_open/preprocessings/signal_processing.py#L52
    '''
    b, a = signal.iirfilter(order, [ff * 2. / fs for ff in frequency_band], btype='bandpass', ftype='butter')
    y = signal.lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff_fs, fs, order=1):
    b, a = butter(order, cutoff_fs, btype='low',fs=fs)
    y = signal.filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff_fs, fs, order=1):
    b, a = butter(order, cutoff_fs, btype='high',fs=fs)
    y = signal.filtfilt(b, a, data)
    return y

def clip_signal(data, clipping_amplitude=500):
    data = np.clip(data, -1*clipping_amplitude, clipping_amplitude)
    return data

def preprocess_eeg(data, fs):
    data = notch_filter(data, fs, stop_fs=60, Q=12)
    data = notch_filter(data, fs, stop_fs=50, Q=10)
    data = notch_filter(data, fs, stop_fs=25, Q=5)
    data = iir_bandpass_filter(data, fs, frequency_band=[1,20], order=2)
    return data

def preprocess_eog(data, fs):
    data = notch_filter(data, fs, stop_fs=60, Q=12)
    data = notch_filter(data, fs, stop_fs=50, Q=10)
    data = notch_filter(data, fs, stop_fs=25, Q=5)
    data = iir_bandpass_filter(data, fs, frequency_band=[0.3,10], order=2)
    return data

def preprocess_emg(data, fs):
    data = notch_filter(data, fs, stop_fs=60, Q=12)
    data = notch_filter(data, fs, stop_fs=50, Q=10)
    data = notch_filter(data, fs, stop_fs=25, Q=5)
    data = butter_highpass_filter(data, 10, fs)
    return data

def filter_sos(data, filterType, cutoffFreq, fs, order, notchWidth=None, zi=None):
    if filterType == 'lowpass':
        sos = butter(order, cutoffFreq, fs=fs, btype='low', analog=False, output='sos')
    elif filterType == 'highpass':
        sos = butter(order, cutoffFreq, fs=fs, btype='high', analog=False, output='sos')
    elif filterType == 'bandpass':
        sos = butter(order, cutoffFreq, fs=fs, btype='band', analog=False, output='sos')
    elif filterType == 'bandstop' or filterType == 'notch':
        if filterType == 'notch':
            if notchWidth is None:
                raise ValueError("Please provide notchWidth for the notch filter.")
            low = cutoffFreq - notchWidth / 2
            high = cutoffFreq + notchWidth / 2
            cutoffFreq = [low, high]
        sos = butter(order, cutoffFreq, fs=fs, btype='bandstop', analog=False, output='sos')
    else:
        raise ValueError("Invalid filter type. Choose from 'lowpass', 'highpass', 'bandpass', 'bandstop', or 'notch'.")

    if zi is None:
        zi = sosfilt_zi(sos)

    filtered_data, zo = sosfilt(sos, data, zi=zi)

    return filtered_data, zo


def filter_iir(data, filter_type, cutoff_freq, fs, order=4, notch_width=None, zi=None):
    if filter_type == 'lowpass':
        b, a = iirfilter(N=order, Wn=cutoff_freq/(fs/2), btype='low', ftype='butter', output='ba')
    elif filter_type == 'highpass':
        b, a = iirfilter(N=order, Wn=cutoff_freq/(fs/2), btype='high', ftype='butter', output='ba')
    elif filter_type == 'bandpass':
        b, a = iirfilter(N=order, Wn=[c/ (fs/2) for c in cutoff_freq], btype='band', ftype='butter', output='ba')
    elif filter_type == 'bandstop' or filter_type == 'notch':
        if filter_type == 'notch':
            if notch_width is None:
                raise ValueError("Please provide notchWidth for the notch filter.")
            low = cutoff_freq - notch_width / 2
            high = cutoff_freq + notch_width / 2
            cutoff_freq = [low, high]
        b, a = iirfilter(N=order, Wn=[c/ (fs/2) for c in cutoff_freq], btype='bandstop', ftype='butter', output='ba')
    else:
        raise ValueError("Invalid filter type. Choose from 'lowpass', 'highpass', 'bandpass', 'bandstop', or 'notch'.")

    if zi is not None:
        filtered_data = filtfilt(b, a, data, zi=zi)
    else:
        filtered_data = filtfilt(b, a, data)
    return filtered_data

def filter_eeg(data, fs, bandpass=[1,50], sos=True):
    for i in range(data.shape[0]):
        if sos:
            filtered, _ = filter_sos(data[i], 'bandpass', bandpass, fs, 4)
            filtered, _ = filter_sos(filtered, 'bandstop', [49,51], fs, 2, notchWidth=4)
            data[i], _ = filter_sos(filtered, 'bandstop', [59,61], fs, 2, notchWidth=4)
        else:
            filtered = filter_iir(data[i], 'bandpass', bandpass, fs, 4)
            filtered = filter_iir(filtered, 'notch', 50.0, fs, 2, notch_width=4)
            data[i] = filter_iir(filtered, 'notch', 60.0, fs, 2, notch_width=4)
    
    return data

def filter_ecg(data, fs, bandpass=[5,50], sos=True):
    if sos:
        filtered, _ = filter_sos(data, 'bandpass', bandpass, fs, 4)
        filtered, _ = filter_sos(filtered, 'notch', 50.0, fs, 2, notchWidth=4)
        filtered, _ = filter_sos(filtered, 'notch', 60.0, fs, 2, notchWidth=4)
    else:
        filtered = filter_iir(data, 'bandpass', bandpass, fs, 4)
        filtered = filter_iir(filtered, 'notch', 50.0, fs, 2, notch_width=4)
        filtered = filter_iir(filtered, 'notch', 60.0, fs, 2, notch_width=4)
    
    b, a = signal.butter(4, 50/(fs/2), 'low')
    tempf = signal.filtfilt(b,a, filtered)
    tempf = signal.filtfilt(b,a, filtered)

    nyq_rate = fs/ 2.0
    width = 5.0/nyq_rate
    ripple_db = 60.0
    O, beta = signal.kaiserord(ripple_db, width)
    cutoff_hz = 4.0
    taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
    filtered = signal.lfilter(taps, 1.0, tempf)
    
    return filtered
