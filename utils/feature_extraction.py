
import numpy as np
import pywt
from scipy.signal import welch

def calculate_psd(eeg_data, fs):
    """
    Calculate the Power Spectral Density (PSD) for each channel using Welch's method.
    """
    psd = []
    freqs = None
    
    for channel in eeg_data:
        freq, p = welch(channel, fs, nperseg=512)
        psd.append(p)
        freqs = freq
    
    psd = np.array(psd)
    return freqs, psd

def extract_average_band_power(psd, freqs, bands):
    """
    Extract average band power for specified frequency bands.
    """
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        band_power = psd[:, idx_band].mean(axis=1)
        band_powers[band_name] = band_power
    return band_powers

def extract_dwt_features(eeg_data, wavelet='db4', level=4):
    """
    Extract features from EEG data using Discrete Wavelet Transform (DWT).
    """
    features = []
    
    for channel_data in eeg_data:
        coeffs = pywt.wavedec(channel_data, wavelet, level=level)
        channel_features = []
        for coeff in coeffs:
            channel_features.append(np.mean(coeff))
            channel_features.append(np.std(coeff))
        features.append(channel_features)
    
    return features

def calculate_entropy(coeffs):
    """
    Calculate Shannon entropy for each set of DWT coefficients.
    """
    entropy = []
    for coeff in coeffs:
        entropy.append(shannon_entropy(coeff))
    return entropy

def shannon_entropy(data):
    """
    Calculate the Shannon entropy of the input data.
    """
    prob_data = data ** 2
    prob_data /= np.sum(prob_data)  # normalize to form a probability distribution
    prob_data = prob_data[np.nonzero(prob_data)]  # remove zeros
    return -np.sum(prob_data * np.log2(prob_data))

# Placeholder functions for indices calculation
# Replace with the actual formulas from your literature/research
def compute_aw_index(psd, freqs):
    return np.zeros(psd.shape[0])  # Placeholder

def compute_choice_index(psd, freqs):
    return np.zeros(psd.shape[0])  # Placeholder

def compute_effort_index(psd, freqs):
    return np.zeros(psd.shape[0])  # Placeholder

def compute_valence_indices(psd, freqs):
    return {
        'vamv_valence': np.zeros(psd.shape[0]),
        'kirk_valence': np.zeros(psd.shape[0]),
        'ram12_valence': np.zeros(psd.shape[0]),
        'ram15_valence': np.zeros(psd.shape[0])
    }  # Placeholder
