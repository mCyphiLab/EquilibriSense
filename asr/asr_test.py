import numpy as np
from clean_drifts import clean_drifts_iir 
from clean_flatlines import clean_flatlines
from asr_calibrate import asr_calibrate
from asr_process import asr_process
from clean_channels_nolocs import clean_channels_nolocs
from clean_artifacts import clean_artifacts, clean_artifacts_realtime
from clean_windows import clean_windows
from clean_asr import clean_asr
import pytest
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import correlate
import seaborn as sns
from asr_utils import geometric_median, block_geometric_median, sphericalSplineInterpolate, interpMx,\
    calc_projector, randsample, mad, design_yulewalk_filter, fit_eeg_distribution, moving_average
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from numpy.fft import fft, fftfreq
import glob
import pandas as pd

def correlation_coefficient(signal1, signal2):
    return np.corrcoef(signal1, signal2)[0, 1]

def mean_squared_error(signal1, signal2):
    return np.mean((signal1 - signal2)**2)

def root_mean_square_error(signal1, signal2):
    return np.sqrt(mean_squared_error(signal1, signal2))
                   
def mean_absolute_error(signal1, signal2):
    return np.mean(np.abs(signal1 - signal2))

def normalized_cross_correlation(signal1, signal2):
    normed_correlation = correlate(signal1, signal2)
    
def load_matlab_output(filename):
    """
    Load the output of the Matlab clean_drifts function from a .mat file.
    """
    mat = scipy.io.loadmat(filename)
    return mat['output_signal']

def load_matlab_output_complete(filename):
    """
    Load the output of the Matlab clean_drifts function from a .mat file.
    """
    mat = scipy.io.loadmat(filename)
    return mat


##################################################################################
##################################################################################
########################### Sub function test cases
##################################################################################
##################################################################################



##################################################################################
##################################################################################
########################### Main function test cases
##################################################################################
##################################################################################

# Clean_drifts test case
def test_clean_drifts(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T,
        'srate': 250,
        'etc':None
    }
    
    # Load the results after cleaning drifts first
    mat_clean_drifts = load_matlab_output('./data/output_clean_drifts.mat')
    # print(input_signal['data'].shape)
    python_clean_drifts = clean_drifts_iir(input_signal)['data'].T
    
    # Plot the correlation heatmap
    num_channels = mat_clean_drifts.shape[0]

    if visualize:
        correlation_matrix = np.zeros((num_channels, num_channels))

        for i in range(num_channels):
            for j in range(num_channels):
                correlation_matrix[i, j] = np.corrcoef(mat_clean_drifts[i, :], python_clean_drifts[j, :])[0, 1]
                
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap between MATLAB and Python Outputs')
        plt.xlabel('Python Clean Drifts Channels')
        plt.ylabel('MATLAB Clean Drifts Channels')
        
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 8))
        
        # Loop through each channel and plot the signals
        for i in range(7):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Plot the signals after filtering
            ax.plot(mat_clean_drifts[i, :], label='MATLAB')
            ax.plot(python_clean_drifts[i, :], label='Python')
            
            # Customize the subplot
            ax.set_title(f'Channel {i + 1}')
            ax.legend()
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
        
        # Add some space between the subplots for clarity
        plt.tight_layout()
        
        # Show the plots
        plt.show()
    

    correlation_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        correlation_matrix[i, i] = np.corrcoef(mat_clean_drifts[i, :], python_clean_drifts[i, :])[0, 1]
    
    # Calculate the mean of the diagonal (correlation of corresponding channels)
    mean_correlation = np.mean(np.diag(correlation_matrix))

    assert mean_correlation > 0.95, f'Mean correlation {mean_correlation:.4f} < 0.95'


def clean_flatlines():
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T,
        'srate': 250,
        'etc':None
    }
    
    # Load the results after cleaning drifts first
    mat_clean_flatlines = load_matlab_output('./data/output_clean_flatlines.mat')
    python_clean_flatlines = clean_flatlines(input_signal)['data']
    assert  mat_clean_flatlines.shape == python_clean_flatlines.shape


def test_asr_calibrate():
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data_new.mat')['eeg_filtered'].T,
        'srate': 250,
        'etc':None
    }
    
    matlab_state_M = load_matlab_output_complete('./data/output_asr_calibrate_M.mat')['out_M']
    matlab_state_T = load_matlab_output_complete('./data/output_asr_calibrate_T.mat')['out_T']
    matlab_state_A = load_matlab_output_complete('./data/output_asr_calibrate_A.mat')['out_A']
    matlab_state_B = load_matlab_output_complete('./data/output_asr_calibrate_B.mat')['out_B']


    python_state = asr_calibrate(input_signal['data'], input_signal['srate'])    
    error_M = root_mean_square_error(python_state['M'], matlab_state_M)
    error_T = root_mean_square_error(abs(python_state['T']), abs(matlab_state_T))
    error_A = root_mean_square_error(python_state['A'], matlab_state_A)
    error_B = root_mean_square_error(python_state['B'], matlab_state_B)
    
    # print(matlab_state_T)
    # print(python_state['T'])

    print('error M: ', error_M)
    print('error T: ', error_T)
    print('error A: ', error_A)
    print('error B: ', error_B)
        
    assert  error_M <= 0.1
    assert  error_T <= 10
    assert  error_A <= 0.1
    assert  error_B <= 0.1


def test_asr_process(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data_new.mat')['eeg_filtered'].T,
        'srate': 250,
        'etc':None
    }
    
    python_state = asr_calibrate(input_signal['data'], input_signal['srate'])
    matlab_state = load_matlab_output_complete('./data/output_asr_calibrate.mat')['out']
        
    M = matlab_state[0, 0]['M']
    T = matlab_state[0, 0]['T']
    B = matlab_state[0, 0]['B']
    A = matlab_state[0, 0]['A']
    cov = matlab_state[0, 0]['cov']
    carry = matlab_state[0, 0]['carry']
    iir = matlab_state[0, 0]['iir']
    last_R = matlab_state[0, 0]['last_R']
    last_trivial = matlab_state[0, 0]['last_trivial']
        
    matlab_state = {
        'M': M.T,
        'T': T,
        'B': B.T.reshape(-1),
        'A': A.T.reshape(-1),
        'cov': None,
        'carry': None,
        'iir': iir.T,
        'last_R': None,
        'last_trivial': True
    }
    
    python_clean_process, _ = asr_process(input_signal['data'], input_signal['srate'], matlab_state)
    matlab_clean_process = load_matlab_output('./data/output_asr_process.mat')


    num_channels = python_clean_process.shape[0]

    if visualize:
        #Plot the correlation heatmap
        correlation_matrix = np.zeros((num_channels, num_channels))

        for i in range(num_channels):
            for j in range(num_channels):
                correlation_matrix[i, j] = np.corrcoef(matlab_clean_process[i, :], python_clean_process[j, :])[0, 1]
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap between MATLAB and Python Outputs')
        plt.xlabel('Python Clean Process Channels')
        plt.ylabel('MATLAB Clean Process Channels')
        
        # Create 2x4 subplots
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 8))
        
        # Loop through each channel and plot the signals
        for i in range(7):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Plot the signals after filtering
            ax.plot(matlab_clean_process[i, :], label='MATLAB')
            ax.plot(python_clean_process[i, :], label='Python')
            
            # Customize the subplot
            ax.set_title(f'Channel {i + 1}')
            ax.legend()
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
        
        # Add some space between the subplots for clarity
        plt.tight_layout()
        plt.show()
        
    # Do the assert for test case running
    correlation_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        correlation_matrix[i, i] = np.corrcoef(matlab_clean_process[i, :], python_clean_process[i, :])[0, 1]
    
    # Calculate the mean of the diagonal (correlation of corresponding channels)
    mean_correlation = np.mean(np.diag(correlation_matrix))

    assert mean_correlation > 0.95, f'Mean correlation {mean_correlation:.4f} < 0.95'
    
    
def test_clean_channels_nolocs(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T,
        'srate': 250,
        'etc': {
            'clean_channel_mask': np.ones(scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T.shape[0], dtype=bool) 
        },
    }
    
    matlab_clean_channels_nolocs_signal = load_matlab_output_complete('./data/output_clean_channels_nolocs_signal.mat')['log_signal']
    matlab_clean_channels_nolocs_removed_channel = load_matlab_output_complete('./data/output_clean_channels_nolocs_removed_channel.mat')['removed_array'].reshape(-1)
    matlab_clean_channels_nolocs_removed_channel = np.array(matlab_clean_channels_nolocs_removed_channel).astype(bool)

    python_clean_channels_nolocs_signal, python_clean_channels_nolocs_removed_channel = clean_channels_nolocs(input_signal, linenoise_aware=False)
    python_clean_channels_nolocs_signal = python_clean_channels_nolocs_signal['data']

    if visualize:
        #Plot the correlation heatmap
        num_channels = matlab_clean_channels_nolocs_signal.shape[0]
        correlation_matrix = np.zeros((num_channels, num_channels))

        for i in range(num_channels):
            for j in range(num_channels):
                correlation_matrix[i, j] = np.corrcoef(matlab_clean_channels_nolocs_signal[i, :], python_clean_channels_nolocs_signal[j, :])[0, 1]
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap between MATLAB and Python Outputs')
        plt.xlabel('Python Clean channels Channels')
        plt.ylabel('MATLAB Clean channels Channels')
        
        # Create 2x4 subplots
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 8))
        
        # Loop through each channel and plot the signals
        for i in range(6):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Plot the signals after filtering
            ax.plot(matlab_clean_channels_nolocs_signal[i, :], label='MATLAB')
            ax.plot(python_clean_channels_nolocs_signal[i, :], label='Python')
            
            # Customize the subplot
            ax.set_title(f'Channel {i + 1}')
            ax.legend()
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
        
        # Add some space between the subplots for clarity
        plt.tight_layout()
        plt.show()
        

    assert np.array_equal(matlab_clean_channels_nolocs_removed_channel, python_clean_channels_nolocs_removed_channel)
    assert python_clean_channels_nolocs_signal.shape == matlab_clean_channels_nolocs_signal.shape
    

def test_clean_windows(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data_new.mat')['eeg_filtered'].T,
        'srate': 250,
        'etc': {
            'clean_channel_mask': np.ones(scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T.shape[0], dtype=bool),
            # 'clean_sample_mask': np.ones(scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T.shape[0], dtype=bool),
        },
    }

    matlab_clean_windows_signal = load_matlab_output_complete('./data/output_clean_windows_signal.mat')['log_signal']
    matlab_clean_windows_removed_windows = load_matlab_output_complete('./data/output_clean_windows_removed.mat')['removed_windows'].reshape(-1)
    matlab_clean_windows_removed_windows = np.array(matlab_clean_windows_removed_windows).astype(bool)
    
    python_clean_windows_signal, python_clean_windows_removed_windows = clean_windows(input_signal, 0.25, [-np.inf, 7])
    python_clean_windows_signal = python_clean_windows_signal['data']
    
    num_channels = matlab_clean_windows_signal.shape[0]

    if visualize:
        #Plot the correlation heatmap
        correlation_matrix = np.zeros((num_channels, num_channels))

        for i in range(num_channels):
            for j in range(num_channels):
                correlation_matrix[i, j] = np.corrcoef(matlab_clean_windows_signal[i, :], python_clean_windows_signal[j, :])[0, 1]
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap between MATLAB and Python Outputs')
        plt.xlabel('Python Clean Windows Channels')
        plt.ylabel('MATLAB Clean Windows Channels')
        
        # Create 2x4 subplots
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 8))
        
        # Loop through each channel and plot the signals
        for i in range(num_channels):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Plot the signals after filtering
            ax.plot(matlab_clean_windows_signal[i, :], label='MATLAB')
            ax.plot(python_clean_windows_signal[i, :], label='Python')
            
            # Customize the subplot
            ax.set_title(f'Channel {i + 1}')
            ax.legend()
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
        
        # Add some space between the subplots for clarity
        plt.tight_layout()
        plt.show()
        
    # Do the assert for test case running
    correlation_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        correlation_matrix[i, i] = np.corrcoef(matlab_clean_windows_signal[i, :], python_clean_windows_signal[i, :])[0, 1]
    # Calculate the mean of the diagonal (correlation of corresponding channels)
    mean_correlation = np.mean(np.diag(correlation_matrix))

    assert mean_correlation > 0.95, f'Mean correlation {mean_correlation:.4f} < 0.95'        
    assert np.array_equal(matlab_clean_windows_removed_windows, python_clean_windows_removed_windows)


def test_fit_eeg_distribution(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data_new.mat')['eeg_filtered'].T,
        'srate': 250,
        'etc': {
            'clean_channel_mask': np.ones(scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T.shape[0], dtype=bool),
        },
    }
    
    matlab_mu = load_matlab_output_complete('./data/output_fit_eeg_distribution_mu.mat')['Mu']
    matlab_sig = load_matlab_output_complete('./data/output_fit_eeg_distribution_sig.mat')['Sig']
    
    python_mu, python_sig, _, _ = fit_eeg_distribution(input_signal['data'])
    rms_mu = root_mean_square_error(python_mu, matlab_mu)
    rms_sig = root_mean_square_error(python_sig, matlab_sig)
    
    assert rms_mu <= 0.01
    assert rms_sig <= 0.01


def test_clean_asr(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data_new.mat')['eeg_filtered'].T,
        'srate': 250,
        'nbchan': 7,
        'etc': {
            'clean_channel_mask': np.ones(scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T.shape[0], dtype=bool),
        },
    }

    python_clean_asr_signal = clean_asr(input_signal)
    python_clean_asr_signal = python_clean_asr_signal['data']

    matlab_clean_asr_signal = load_matlab_output_complete('./data/output_clean_asr.mat')['output_data']
    
    # print(matlab_clean_asr_signal.shape)
    # print(python_clean_asr_signal.shape)

    num_channels = matlab_clean_asr_signal.shape[0]

    if visualize:
        #Plot the correlation heatmap
        correlation_matrix = np.zeros((num_channels, num_channels))

        for i in range(num_channels):
            for j in range(num_channels):
                correlation_matrix[i, j] = np.corrcoef(matlab_clean_asr_signal[i, :], python_clean_asr_signal[j, :])[0, 1]
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap between MATLAB and Python Outputs')
        plt.xlabel('Python Clean ASR Channels')
        plt.ylabel('MATLAB Clean ASR Channels')
        
        # Create 2x4 subplots
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 8))
        
        # Loop through each channel and plot the signals
        for i in range(num_channels):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Plot the signals after filtering
            ax.plot(matlab_clean_asr_signal[i, :], label='MATLAB')
            ax.plot(python_clean_asr_signal[i, :], label='Python')
            
            # Customize the subplot
            ax.set_title(f'Channel {i + 1}')
            ax.legend()
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
        
        # Add some space between the subplots for clarity
        plt.tight_layout()
        plt.show()
        
    # Do the assert for test case running
    correlation_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        correlation_matrix[i, i] = np.corrcoef(matlab_clean_asr_signal[i, :], python_clean_asr_signal[i, :])[0, 1]
    # Calculate the mean of the diagonal (correlation of corresponding channels)
    mean_correlation = np.mean(np.diag(correlation_matrix))

    assert mean_correlation > 0.95, f'Mean correlation {mean_correlation:.4f} < 0.95'        



def compute_psd(channel_data):
    yf = fft(channel_data)
    xf = fftfreq(len(channel_data), 1 / 250)
    psd = np.abs(yf) ** 2

    desired_freq_mask = (xf >= 0) & (xf <= 30)  # Keep only positive frequencies and up to 30Hz
    return xf[desired_freq_mask], psd[desired_freq_mask]


def test_clean_artifacts(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data_new.mat')['eeg_filtered'].T,
        'srate': 250,
        'nbchan': 7,
        'etc': {
            'clean_channel_mask': np.ones(scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T.shape[0], dtype=bool),
        },
    }
    
    python_clean_artifacts_signal = clean_artifacts(input_signal)
    python_clean_artifacts_signal = python_clean_artifacts_signal['data']

    matlab_clean_artifacts_signal = load_matlab_output_complete('./data/output_clean_artifacts.mat')['output_data']

    num_channels = python_clean_artifacts_signal.shape[0]

    for i in range(num_channels):
        line_data = matlab_clean_artifacts_signal[i, :].copy().astype(np.float64)
        DataFilter.detrend(line_data, DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(line_data, 125, 5, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(line_data, 125, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        matlab_clean_artifacts_signal[i, :] = line_data

    for i in range(num_channels):
        line_data = python_clean_artifacts_signal[i, :].copy().astype(np.float64)
        DataFilter.detrend(line_data, DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(line_data, 125, 5, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(line_data, 125, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        python_clean_artifacts_signal[i, :] = line_data

    if visualize:
        # Plot the correlation heatmap
        correlation_matrix = np.zeros((num_channels, num_channels))

        for i in range(num_channels):
            for j in range(num_channels):
                correlation_matrix[i, j] = np.corrcoef(matlab_clean_artifacts_signal[i, :], python_clean_artifacts_signal[j, :])[0, 1]
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap between MATLAB and Python Outputs')
        plt.xlabel('Python Clean Artifacts Channels')
        plt.ylabel('MATLAB Clean Artifacts Channels')
        
        fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 14))
        
        # Loop through each channel and plot the signals
        for i in range(num_channels):
            # EEG Data
            axes[i, 0].plot(python_clean_artifacts_signal[i, :], label='Python', linewidth=0.5)
            axes[i, 0].plot(matlab_clean_artifacts_signal[i, :], label='Matlab', linewidth=0.5)

            axes[i, 0].set_title(f'Channel {i + 1} EEG')
            axes[i, 0].legend()
            axes[i, 0].set_xlabel('Samples')
            axes[i, 0].set_ylabel('Amplitude')
            
            # PSD
            xf, psd = compute_psd(matlab_clean_artifacts_signal[i, :])
            axes[i, 1].plot(xf, psd, color='b')
            axes[i, 1].set_title(f'Channel {i + 1} PSD')
            axes[i, 1].set_xlabel('Frequency (Hz)')
            axes[i, 1].set_ylabel('PSD')
        
        # fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 14))
        
        # # Plot before processing data
        # before_data = scipy.io.loadmat('./data/eeg_filtered_data_new_no_filter.mat')['eeg_selected'].T
        
        # for i in range(num_channels):
        #     # EEG Data
        #     axes[i, 0].plot(before_data[i, 500:1000], label='EEG', linewidth=0.5)

        #     axes[i, 0].set_title(f'Channel {i + 1} EEG')
        #     axes[i, 0].legend()
        #     axes[i, 0].set_xlabel('Samples')
        #     axes[i, 0].set_ylabel('Amplitude')
            
        #     # PSD
        #     xf, psd = compute_psd(before_data[i, :])
        #     axes[i, 1].plot(xf, psd, color='b')
        #     axes[i, 1].set_title(f'Channel {i + 1} PSD')
        #     axes[i, 1].set_xlabel('Frequency (Hz)')
        #     axes[i, 1].set_ylabel('PSD')
        #     axes[i, 1].set_ylim(0, 5000000000)
        # Add some space between the subplots for clarity
        plt.tight_layout()
        plt.show()
        
        
        
        
def clean_artifacts_realtime(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data_new.mat')['eeg_filtered'].T,
        'srate': 250,
        'nbchan': 7,
        'etc': {
            'clean_channel_mask': np.ones(scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T.shape[0], dtype=bool),
        },
    }
    
    python_clean_artifacts_signal = clean_artifacts_realtime(input_signal)


def plot_std_bar(data, ax, label, color='blue'):
    """
    Plot bar chart of the provided data on the given axes.
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
        
    ax.bar(label, mean_val, color=color, capsize=10, alpha=0.75)
    # ax.set_title(f"Average PSD for '{label}'")
    ax.set_ylabel('PSD (uV^2/Hz)')
    offset = 0.05 * (mean_val + std_val)
    # ax.set_ylim(0, mean_val + std_val + offset)


def test_integrated_asr(visualize=False):
    input_signal = {
        'data': scipy.io.loadmat('./data/eeg_filtered_data_new.mat')['eeg_filtered'].T,
        'srate': 250,
        'nbchan': 7,
        'etc': {
            'clean_channel_mask': np.ones(scipy.io.loadmat('./data/eeg_filtered_data.mat')['eeg_filtered'].T.shape[0], dtype=bool),
        },
    }
    
    pre_asr_data = input_signal['data']
        
    post_asr_data = clean_artifacts(input_signal)
    post_asr_data = post_asr_data['data']    

    num_chans = post_asr_data.shape[0]

    for i in range(num_chans):
        line_data = post_asr_data[i, :].copy().astype(np.float64)
        DataFilter.detrend(line_data, DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(line_data, 125, 5, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(line_data, 125, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        post_asr_data[i, :] = line_data
        
        
        

    channel_names = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz']
    cmap = plt.cm.Set1

    WINDOW_SIZE = 250*2 # 2-second windows
    OVERLAP = int(WINDOW_SIZE * 0.0)  # 50% overlap

    spectro_before = []
    spectro_after = []

    data_len = len(post_asr_data[0])
    nfft = DataFilter.get_nearest_power_of_two(250)

    ############################################################################
    ############################### Before ASR #################################
    ############################################################################
    fig, axs = plt.subplots(4, 2, figsize=(17, 7), sharex=True)
    fig1, axs1 = plt.subplots(4, 2, figsize=(17, 7), sharex=False)

    for i in range(num_chans):
        spectrogram = {'delta':[], 'theta':[], 'alpha':[], 'beta':[], 'gamma':[]}

        process_data = pre_asr_data[i,:]
        for start in range(0, data_len - WINDOW_SIZE, WINDOW_SIZE - OVERLAP):
            window = process_data[start:start + WINDOW_SIZE]
            
            if len(window) < nfft:
                break
                            
            psd = DataFilter.get_psd_welch(window, nfft, nfft // 2, 250, WindowOperations.BLACKMAN_HARRIS.value)
            
            band_power_delta = DataFilter.get_band_power(psd, 2.0, 4.0)
            band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)
            band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)
            band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
            band_power_gamma = DataFilter.get_band_power(psd, 30.0, 50.0)
            
            spectrogram['delta'].append(band_power_delta)
            spectrogram['theta'].append(band_power_theta)
            spectrogram['alpha'].append(band_power_alpha)
            spectrogram['beta'].append(band_power_beta)
            spectrogram['gamma'].append(band_power_gamma)
            
        time_x = np.arange(len(spectrogram['alpha'])) * 2
        
        axs[i // 2, i % 2].plot(time_x, spectrogram['alpha'], label='Alpha', color=cmap(4))
        axs[i // 2, i % 2].plot(time_x, spectrogram['beta'], label='Beta', color=cmap(3))
        axs[i // 2, i % 2].plot(time_x, spectrogram['theta'], label='Theta', color=cmap(2))

        axs[i // 2, i % 2].legend(fontsize=10, loc='upper right')
        axs[i // 2, i % 2].set_ylabel('V**2/Hz',fontsize=12)
        axs[i // 2, i % 2].set_title(f'PSD - {channel_names[i]}',fontsize=12)
        
        plot_std_bar(spectrogram['beta'], axs1[i // 2, i % 2], 'Non ASR')

        axs1[i // 2, i % 2].set_title(f'PSD - Beta {channel_names[i]}',fontsize=12)
        
        axs[3, 0].set_xlabel('Time (s)',fontsize=12)
        axs[3, 1].set_xlabel('Time (s)',fontsize=12)
        
        
    ############################################################################
    ################################ After ASR #################################
    ############################################################################
    fig, axs = plt.subplots(4, 2, figsize=(17, 7), sharex=True)

    for i in range(num_chans):
        spectrogram = {'delta':[], 'theta':[], 'alpha':[], 'beta':[], 'gamma':[]}

        process_data = post_asr_data[i,:]
        for start in range(0, data_len - WINDOW_SIZE, WINDOW_SIZE - OVERLAP):
            window = process_data[start:start + WINDOW_SIZE]
            
            if len(window) < nfft:
                break
                            
            psd = DataFilter.get_psd_welch(window, nfft, nfft // 2, 250, WindowOperations.BLACKMAN_HARRIS.value)
            
            band_power_delta = DataFilter.get_band_power(psd, 2.0, 4.0)
            band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)
            band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)
            band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
            band_power_gamma = DataFilter.get_band_power(psd, 30.0, 50.0)
            
            spectrogram['delta'].append(band_power_delta)
            spectrogram['theta'].append(band_power_theta)
            spectrogram['alpha'].append(band_power_alpha)
            spectrogram['beta'].append(band_power_beta)
            spectrogram['gamma'].append(band_power_gamma)
            
        time_x = np.arange(len(spectrogram['alpha'])) * 2
        
        axs[i // 2, i % 2].plot(time_x, spectrogram['alpha'], label='Alpha', color=cmap(4))
        axs[i // 2, i % 2].plot(time_x, spectrogram['beta'], label='Beta', color=cmap(3))
        axs[i // 2, i % 2].plot(time_x, spectrogram['theta'], label='Theta', color=cmap(2))

        axs[i // 2, i % 2].legend(fontsize=10, loc='upper right')
        axs[i // 2, i % 2].set_ylabel('V**2/Hz',fontsize=12)
        axs[i // 2, i % 2].set_title(f'PSD - {channel_names[i]}',fontsize=12)

        plot_std_bar(spectrogram['beta'], axs1[i // 2, i % 2], 'Applied ASR', color='red')
        axs1[i // 2, i % 2].set_title(f'PSD - Beta {channel_names[i]}',fontsize=12)
        
    axs[3, 0].set_xlabel('Time (s)',fontsize=12)
    axs[3, 1].set_xlabel('Time (s)',fontsize=12)
    

    ############################################################################
    ################################ After ASR #################################
    ############################################################################
    # Get file names
    # file_names = sorted(glob.glob('./data/logging/Head_Turn_Testing/CONVERTED_*'))

    # # Combine all files into a single DataFrame
    # data = pd.concat([pd.read_csv(file_name, sep='\t') for file_name in file_names])

    # data = data.drop(columns=['index', 'EDA', 'time stamp', 'red led', 'ir led', 'green', 'temperature', 'EDA'])

    # filtered_data_list = []

    # # Bandpass the EEG signals
    # for i in range(7):
    #     process_data = data.iloc[:, i].to_numpy().copy()
    #     DataFilter.detrend(process_data, DetrendOperations.LINEAR.value)
    #     DataFilter.perform_bandpass(process_data, 125, 5, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    #     DataFilter.perform_bandstop(process_data, 125, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    #     filtered_data_list.append(process_data)
        
    
    # filtered_data_list = np.array(filtered_data_list)
    # start_idx = 0*250
    # stop_idx = 36*250
    
    # plot_data = filtered_data_list[:,start_idx:stop_idx]

    # data_len = len(plot_data[0])

    # fig, axs = plt.subplots(4, 2, figsize=(17, 7), sharex=True)

    # for i in range(7):
    #     spectrogram = {'delta':[], 'theta':[], 'alpha':[], 'beta':[], 'gamma':[]}

    #     process_data = plot_data[i,:]
    #     for start in range(0, data_len - WINDOW_SIZE, WINDOW_SIZE - OVERLAP):
    #         window = process_data[start:start + WINDOW_SIZE]
            
    #         if len(window) < nfft:
    #             break
                            
    #         psd = DataFilter.get_psd_welch(window, nfft, nfft // 2, 250, WindowOperations.BLACKMAN_HARRIS.value)
            
    #         band_power_delta = DataFilter.get_band_power(psd, 2.0, 4.0)
    #         band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)
    #         band_power_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)
    #         band_power_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
    #         band_power_gamma = DataFilter.get_band_power(psd, 30.0, 50.0)
            
    #         spectrogram['delta'].append(band_power_delta)
    #         spectrogram['theta'].append(band_power_theta)
    #         spectrogram['alpha'].append(band_power_alpha)
    #         spectrogram['beta'].append(band_power_beta)
    #         spectrogram['gamma'].append(band_power_gamma)
            
    #     spectro_after.append(spectrogram['theta'])

    #     time_x = np.arange(len(spectrogram['alpha'])) * 2
        
    #     axs[i // 2, i % 2].plot(time_x, spectrogram['alpha'], label='Alpha', color=cmap(4))
    #     axs[i // 2, i % 2].plot(time_x, spectrogram['beta'], label='Beta', color=cmap(3))
    #     axs[i // 2, i % 2].plot(time_x, spectrogram['theta'], label='Theta', color=cmap(2))

    #     axs[i // 2, i % 2].legend(fontsize=10, loc='upper right')
    #     axs[i // 2, i % 2].set_ylabel('V**2/Hz',fontsize=12)
    #     axs[i // 2, i % 2].set_title(f'PSD - {channel_names[i]}',fontsize=12)

    #     plot_std_bar(spectrogram['beta'], axs1[i // 2, i % 2], 'Non-noisy data', color='red')
    #     axs1[i // 2, i % 2].set_title(f'PSD - Beta {channel_names[i]}',fontsize=12)
        
    # axs[3, 0].set_xlabel('Time (s)',fontsize=12)
    # axs[3, 1].set_xlabel('Time (s)',fontsize=12)

    plt.show()

if __name__ == "__main__":
    test_integrated_asr(True)
        
    