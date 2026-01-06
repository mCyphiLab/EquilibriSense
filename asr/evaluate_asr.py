import glob
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import iirfilter, filtfilt
import matplotlib.pyplot as plt
from .clean_artifacts import clean_artifacts
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


# Function to plot EEG data
def plot_eeg(data, title, channel_names):
    fig, axs = plt.subplots(4, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(title)
    for i, ax in enumerate(axs.flat):
        ax.plot(data[i, :])
        ax.set_title(channel_names[i])
        ax.grid(True)
    plt.xlabel('Time (samples)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def add_noise_to_pattern(spiky_pattern, noise_level=1e-7):
    noise = np.random.normal(0, noise_level, spiky_pattern.shape)
    noisy_spiky_pattern = spiky_pattern + noise
    return noisy_spiky_pattern

def filter_eeg_channel(channel_data, fs, cutoff, order=4):
    """
    Apply bandpass and notch filters to a single channel of EEG data.
    
    Parameters:
        channel_data (array): 1-D array containing EEG data for a single channel.
        fs (int): Sampling frequency.
        cutoff (list): List containing the low and high cutoff frequencies for bandpass filtering.
        order (int): Order of the IIR filter.
        
    Returns:
        array: Filtered EEG data for the single channel.
    """
    def apply_iir_filter(data, fs, cutoff, btype, order=4, ftype='butter'):
        b, a = iirfilter(N=order, Wn=np.array(cutoff)/(fs/2), btype=btype, ftype=ftype)
        return filtfilt(b, a, data)
    
    # Bandpass filter
    filtered_data = apply_iir_filter(channel_data, fs, cutoff, 'band', order)
    
    # Notch filter at 50 Hz
    filtered_data = apply_iir_filter(filtered_data, fs, [48, 52], 'bandstop', order)
    
    # Notch filter at 60 Hz
    filtered_data = apply_iir_filter(filtered_data, fs, [58, 62], 'bandstop', order)
    
    return filtered_data

def grid_search_asr(original_clean_data, mixed_signal, original_noisy_data, params):
    best_corr = -np.inf  # Initialize with negative infinity
    best_params = None  # Best parameters will be stored here
    
    # Loop through the parameter grid
    for windowlen in params['windowlen']:
        for stepsize in params['stepsize']:
            for maxdims in params['maxdims']:
                input_signal = {
                    'data': mixed_signal,
                    'srate': 250,
                    'nbchan': 8,
                    'etc': {
                        'clean_channel_mask': np.ones(mixed_signal.shape[0], dtype=bool),
                    },
                }
                
                artifacts, cleaned_data = clean_artifacts(input_signal, spiky_pattern_8x, windowlen, stepsize, maxdims)
                artifacts = artifacts['data']    
                
                # Compute the correlation between original clean and cleaned signals
                correlations = []
                for ch in range(original_clean_data.shape[0]):
                    corr, _ = scipy.stats.pearsonr(original_clean_data[ch, :], cleaned_data[ch, :])
                    correlations.append(corr)
                
                avg_corr = np.mean(correlations)
                
                # Update the best parameters if the current correlation is better
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    best_params = {
                        'windowlen': windowlen,
                        'stepsize': stepsize,
                        'maxdims': maxdims
                    }
                    
                print(f"Parameters: Window Length: {windowlen}, Step Size: {stepsize}, Max Dims: {maxdims}, Average Correlation: {avg_corr}")
                
    return best_params, best_corr

def add_noise(clean_signal, noise, snr):
    """
    Add scaled noise to the clean signal.
    
    Parameters:
    - clean_signal: clean EEG signal
    - noise: noise to be added
    - snr: desired signal-to-noise ratio
    """
    
    # Ensure the noise is not zero, and scale it
    noise = noise / np.std(noise)
    
    # Calculate the scaling factor based on desired SNR
    k = 10 ** (-snr / 20) * np.std(clean_signal)
    
    # Add scaled noise to the clean signal
    contaminated_signal = clean_signal + k * noise
    
    return contaminated_signal

##################################################################################
############################# Initialize everything ##############################
##################################################################################
# Parameters
fs = 250  
duration = 10
frequency = 4

t = np.arange(0, duration, 1/fs)
head_shaking_pattern = np.sin(2 * np.pi * frequency * t)
noise_amplitude = 0.2
spiky_pattern = head_shaking_pattern + noise_amplitude * np.random.randn(len(t))
# spiky_pattern_1 = 5 * np.random.randn(len(t))

# Randomly select an amplitude multiplier within a plausible range
amplitude_multiplier = np.random.uniform(200, 400)  # Assuming motion artifact amplitude is between 0.5 to 2 times the original amplitude
spiky_pattern *= amplitude_multiplier

file_names = sorted(glob.glob('./data/logging/EDA_Noise_082823/CONVERTED_*'))
data = pd.concat([pd.read_csv(file_name, sep='\t') for file_name in file_names])
# Drop unneeded data
data = data.drop(columns=['index', 'EDA', 'time stamp', 'red led', 'ir led', 'green', 'temperature', 'EDA'])
filtered_data_list = []
num_chans = 8

# Bandpass the EEG signals
for i in range(num_chans):
    process_data = data.iloc[:, i].to_numpy().copy()
    # DataFilter.detrend(process_data, DetrendOperations.LINEAR.value)
    # DataFilter.perform_bandpass(process_data, fs, 0.5, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    # DataFilter.perform_bandstop(process_data, fs, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    # DataFilter.perform_bandstop(process_data, fs, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    process_data = filter_eeg_channel(process_data, fs, [0.5, 50], order=4)
    filtered_data_list.append(process_data)
    
filtered_data_list = np.array(filtered_data_list)
channel_names = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'O1', 'O2']

start_idx = 30*fs
end_idx = start_idx + duration*fs

filtered_data_list = filtered_data_list[:,start_idx:end_idx]

clean_signal = filtered_data_list[6,:].copy()
clean_signals = filtered_data_list.copy()

# Expand the spiky pattern to all channels
spiky_pattern_8x = np.tile(spiky_pattern, (num_chans, 1))

# Adding slight variations to each row
for i in range(num_chans):
    spiky_pattern_8x[i] = add_noise_to_pattern(spiky_pattern_8x[i])

# Adding the varied spiky patterns to the filtered data
for i in range(num_chans):
    filtered_data_list[i] += spiky_pattern_8x[i]

noisy_signal = filtered_data_list.copy()

###################################################################################
############################# Applying ASR algorithm ##############################
###################################################################################
input_signal = {
    'data': noisy_signal,
    'srate': fs,
    'nbchan': num_chans,
    'etc': {
        'clean_channel_mask': np.ones(noisy_signal.shape[0], dtype=bool),
    },
}

param_grid = {
    'windowlen': np.arange(0.1, 2.1, 0.1),
    'stepsize': [8, 16, 24, 32, 40, 48, 56, 64, 72], 
    'maxdims': np.arange(0.1, 1.1, 0.02)  
}

best_params, best_corr = grid_search_asr(clean_signals, noisy_signal, spiky_pattern_8x, param_grid)

# best_corr = avg_corr
# best_params = {
#     'windowlen': windowlen,
#     'stepsize': stepsize,
#     'maxdims': maxdims
# }


artifacts, clean_data = clean_artifacts(input_signal, spiky_pattern_8x, best_params['windowlen'], best_params['stepsize'], best_params['maxdims'])
artifacts = artifacts['data']    


###################################################################################
################################## Normalized LMS #################################
###################################################################################
import padasip as pa

lms_filter = pa.filters.FilterRLS(n=1, mu=0.4, w="random")
# lms_filter = pa.filters.FilterVSLMS_Mathews(n=1, mu=0.1, ro=0.001, w="random")

MA_desired_signal = []
MA_e = []
for i in range(num_chans):
    desired_signal, e, w = lms_filter.run(spiky_pattern_8x[i], artifacts[i].reshape(len(clean_data[i]), 1))
    MA_desired_signal.append(noisy_signal[i] - desired_signal)
    MA_e.append(e)

###################################################################################
################################## Visualization ##################################
###################################################################################

fig, axs = plt.subplots(4, 2, figsize=(15, 20))  # 4 rows, 2 columns, and set figure size

for i in range(8):  # loop through 8 channels
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    ax.plot(t, clean_signals[i], linewidth=0.5, label='Original clean EEG')
    ax.plot(t, MA_desired_signal[i], linewidth=0.5, label='EEG after ASR')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title(f'Signal comparison - {channel_names[i]}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    

fig, axs = plt.subplots(4, 2, figsize=(15, 20))  # 4 rows, 2 columns, and set figure size

for i in range(8):  # loop through 8 channels
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    ax.plot(t, spiky_pattern_8x[i], linewidth=0.5, label='Original Motion Artifacts')
    ax.plot(t, artifacts[i], linewidth=0.5, label='Extracted Motion Artifacts after ASR')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title(f'Motion artifacts comparison - {channel_names[i]}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

#Plot the correlation heatmap
correlation_matrix = np.zeros((num_chans, num_chans))

for i in range(num_chans):
    for j in range(num_chans):
        correlation_matrix[i, j] = np.corrcoef(clean_signals[i], MA_desired_signal[j])[0, 1]


plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap between original clean EEG and EEG after ASR')
plt.xlabel('Original clean EEG')
plt.ylabel('EEG after ASR')

plt.tight_layout()  # Adjust the spacing between plots for clarity
plt.legend()
plt.show()
