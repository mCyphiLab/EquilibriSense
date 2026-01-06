import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import glob
import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import scipy.fftpack    
import scipy.signal as signal
from scipy.signal import resample
from utils.signal_processing import filter_sos, filter_iir
from scipy.signal import find_peaks
import math
import neurokit2 as nk
from utils.channel_selection import is_epoch_signal_bad

def Resample(hb, samples):
    samples_per_segment = samples // len(hb)

    # Create an array to hold the distributed heart rate values
    distributed_ecg_hr = np.zeros(samples)

    # Assign each heart rate value to its corresponding segment in the array
    for i in range(len(hb)):
        if i < len(hb) - 1:
            distributed_ecg_hr[i * samples_per_segment:(i + 1) * samples_per_segment] = hb[i]
        else:
            # Make sure to fill up to the total_samples for the last segment
            distributed_ecg_hr[i * samples_per_segment:] = hb[i]

    return distributed_ecg_hr


def calculate_heart_rate(ecg_signal, fs):
    # Detect peaks in the segment
    threshold = np.max(ecg_signal) * 0.4  # Simple threshold based on segment mean
    peaks, _ = signal.find_peaks(ecg_signal, height=threshold, distance=fs/3)
    
    # Calculate RR intervals and heart rate if there are enough peaks
    if len(peaks) > 1:
        RR_intervals = np.diff(peaks) / fs
        average_RR_interval = np.mean(RR_intervals)
        heart_rate = 60 / average_RR_interval
    else:
        # heart_rates.append(None)  # Indicate that heart rate couldn't be calculated for this segment
        heart_rate = 0
    
    return heart_rate

def compute_vector_sum(accel_x_filtered=None, accel_y_filtered=None, accel_z_filtered=None):
    """Compute and return vector sum of accelerometer data."""        
    return np.sqrt(accel_x_filtered**2 + accel_y_filtered**2 + accel_z_filtered**2).tolist()

# Custom parameters
fs = 250
window_length_sec = 10
window_samples = window_length_sec * fs

# Initialize directories and data frames
dir_name = '1713994149_Saad'

directories = {
    f'{dir_name}/1713994149_BASE_LINE': 'BASE_LINE',
    f'{dir_name}/1713994824_VOR_1HZ': 'VOR_1HZ',
    f'{dir_name}/1713994964_VOR_1HZ_REST': 'VOR_1HZ_REST',
    f'{dir_name}/1713995471_VOR_2HZ': 'VOR_2HZ',
    f'{dir_name}/1713995610_VOR_2HZ_REST': 'VOR_2HZ_REST',
    f'{dir_name}/1713996271_VOR_3HZ': 'VOR_3HZ',
    f'{dir_name}/1713996412_VOR_3HZ_REST': 'VOR_3HZ_REST',
    f'{dir_name}/1713996958_VOR_4HZ': 'VOR_4HZ',
    f'{dir_name}/1713997104_VOR_4HZ_REST': 'VOR_4HZ_REST'
}

data_frames = {}
length = {}

global_start_idx = 0
for dir_key, label in directories.items():
    file_pattern = f'./data/logging/{dir_key}/CONVERTED_*'
    file_names = sorted(glob.glob(file_pattern))
    data_frames[label] = pd.concat([pd.read_csv(file_name, sep='\t') for file_name in file_names])
    data_frames[label] = data_frames[label].drop(columns=['index', 'time stamp'])
    length[label] = len(data_frames[label])
    if label == 'VOR_4HZ':
        vor_4hz_start = global_start_idx
        vor_4hz_end = global_start_idx + len(data_frames[label])
    global_start_idx += len(data_frames[label])
    
data = pd.concat(data_frames.values())

# Channel names and labels
channel_names = ['O1', 'O2', 'C3', 'C4', 'H-EOG', 'V-EOG', 'ECG', 'EDA', 'Heart rate', 'accel']    
channel_labels = ['uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'V/1024', 'BPM', 'g (1g = 9.8m/s^2)']

# Preprocess and filter data
filtered_data = []
raw_data = []
before_processing_data = []
for i in range(data.shape[1]):
    column_data = data.iloc[:, i].to_numpy().copy()
    raw_data.append(column_data)
    # 7 channels
    if i < 6:
        column1 = filter_iir(column_data, 'bandpass', [1, 50.0], fs, 4)
        column1 = filter_iir(column1, 'notch', 50.0, fs, 2, notch_width=4)
        column1 = filter_iir(column1, 'notch', 60.0, fs, 2, notch_width=4)
                
        column_data = filter_iir(column_data, 'bandpass', [5, 50.0], fs, 4)
        column_data = filter_iir(column_data, 'notch', 50.0, fs, 2, notch_width=4)
        column_data = filter_iir(column_data, 'notch', 60.0, fs, 2, notch_width=4)
    if i == 6:
        column1 = filter_iir(column_data, 'bandpass', [1, 50.0], fs, 4)
        column1 = filter_iir(column1, 'notch', 50.0, fs, 2, notch_width=4)
        column1 = filter_iir(column1, 'notch', 60.0, fs, 2, notch_width=4)
        
        column_data = filter_iir(column_data, 'bandpass', [5, 50.0], fs, 4)
        column_data = filter_iir(column_data, 'notch', 50.0, fs, 2, notch_width=4)
        column_data = filter_iir(column_data, 'notch', 60.0, fs, 2, notch_width=4)
        
        ############################################
        b, a = signal.butter(4, 50/(fs/2), 'low')

        tempf = signal.filtfilt(b,a, column_data)
        yff = scipy.fftpack.fft(tempf)

        nyq_rate = fs/ 2.0
        width = 5.0/nyq_rate
        ripple_db = 60.0
        O, beta = signal.kaiserord(ripple_db, width)
        cutoff_hz = 4.0
        taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
        column_data = signal.lfilter(taps, 1.0, tempf)

        ############################################                
        b, a = signal.butter(4, 50/(fs/2), 'low')
        tempf = signal.filtfilt(b,a, column1)
        yff = scipy.fftpack.fft(tempf)

        nyq_rate = fs/ 2.0
        width = 5.0/nyq_rate
        ripple_db = 60.0
        O, beta = signal.kaiserord(ripple_db, width)
        cutoff_hz = 4.0
        taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
        column1 = signal.lfilter(taps, 1.0, tempf)      
        

    filtered_data.append(column_data)
    before_processing_data.append(column1)
    
vector_sum = compute_vector_sum(filtered_data[8], filtered_data[9], filtered_data[10])
filtered_data.append(vector_sum)
before_processing_data.append(vector_sum)
    
# Inverse the ECG signal to do the peak detection easily
filtered_data = np.array(filtered_data)
# filtered_data[6]*=-1.0   
raw_data = np.array(raw_data)
    
asr_data = filtered_data.copy()
before_processing_data = np.array(before_processing_data)

# Apply ASR
from meegkit.asr import ASR
for i in range(6):
    asr_data[i] = filter_iir(asr_data[i], 'bandpass', [4, 40.0], fs, 4)

calib = asr_data[:6, 510*fs:570*fs]
for i in range(3):
    asr = ASR(sfreq=fs, method='rieman', estimator='lwf', win_overlap=0.7, cutoff=20)   
    _, sample_mask = asr.fit(calib[i*2: i*2+2])        
    window_size = 10*fs

    for idx in range(0, asr_data.shape[1], window_size):     
        bad_data = asr_data[i*2: i*2+2, idx:idx+window_size]        
        clean_data, ma = asr.transform(bad_data)
        asr_data[i*2: i*2+2, idx:idx+window_size] = clean_data

# Select the data from 1713296831_VOR_4HZ for visualization and labeling
vor_4hz_data = before_processing_data[:, vor_4hz_start:vor_4hz_end]
vor_4hz_filtered_data = asr_data[:, vor_4hz_start:vor_4hz_end]

# print(vor_4hz_data.shape, vor_4hz_filtered_data.shape)
# print(data_frames['VOR_4HZ'].index[0], data_frames['VOR_4HZ'].index[-1])
# exit(1)

# Initialize subplot
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.1, wspace=0.4)

fig, axs = plt.subplots(5, 2, figsize=(15, 15), sharex=True)
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.1, wspace=0.4)

# Initialize data lines for plotting
lines = []
x_init = np.linspace(0, window_samples/fs, window_samples)

for i, channel_name in enumerate(channel_names):
    if i < 9:  
        if i == 7:
            eda = np.array(vor_4hz_filtered_data[i, :window_samples])[::250]
            eda = np.interp(np.linspace(0, len(eda), window_samples), np.arange(len(eda)), eda)
            lines.append(axs[i//2, i%2].plot(x_init, eda, label=channel_name, linewidth=0.5)[0])
        elif i == 8:
            ecg = np.interp(np.linspace(0, len(vor_4hz_filtered_data[i-2, :window_samples]), window_samples), np.arange(len(vor_4hz_filtered_data[i-2, :window_samples])), vor_4hz_filtered_data[i-2, :window_samples])
            lines.append(axs[i//2, i%2].plot(x_init, ecg, label=channel_name, linewidth=0.5)[0])
        else:
            lines.append([axs[i//2, i%2].plot(x_init, vor_4hz_filtered_data[i, :window_samples], label=channel_name, linewidth=0.5)[0], axs[i//2, i%2].plot(x_init, vor_4hz_filtered_data[i, :window_samples], label='ASR', linewidth=0.5)[0]])
    else:
        imu_labels = ['Accel X', 'Accel Y', 'Accel Z', 'Vector Sum']
        lines.append([axs[i//2, i%2].plot(x_init, vor_4hz_filtered_data[j-1, :window_samples], label=imu_labels[j-i], linewidth=0.5)[0] for j in range(i, i+4)])

# Update labels
for i, ax in enumerate(axs.flatten()):
    ax.set_title(channel_names[i])
    ax.set_ylabel(channel_labels[i])
    if i == (len(channel_names) - 1) or i == (len(channel_names) - 2):
        ax.set_xlabel('Time (seconds) from start of recording')

# Initialize signal quality labels
signal_quality = np.ones((7, vor_4hz_data.shape[1]), dtype=int)


def update(val):
    start_index = int(slider.val)
    end_index = start_index + window_samples
    x_data = np.linspace(start_index/fs, end_index/fs, end_index-start_index)
    
    for i in range(len(lines)):
        if i < 9: 
            # EDA
            if i == 7:
                eda = filtered_data[i, start_index+vor_4hz_start:end_index+vor_4hz_start][::fs]
                # Calculate the heart rate here
                plot_data = Resample(eda, window_samples)
                
                # Plot the heart rate
                lines[i].set_ydata(plot_data)
                lines[i].set_xdata(x_data)
                axs[i//2, i%2].set_xlim([x_data.min(), x_data.max()])
                axs[i//2, i%2].set_ylim([plot_data.min(), plot_data.max()])   
            # Heart rate
            elif i == 8: 
                if(start_index == 2670*fs):
                    start_index1 = 1570*fs
                    end_index1 = start_index + window_samples
                    ecg = vor_4hz_filtered_data[i-2, start_index1:end_index1]
                    plot_data = 80
                    plot_data = Resample([plot_data], window_samples) 
                else:
                    ecg = vor_4hz_data[i-2, start_index:end_index]               
                    # Calculate the heart rate here
                    # ecg = hp.remove_baseline_wander(ecg, fs)
                    # wd, m = hp.process(hp.scale_data(ecg), fs)
                    # hb = m['bpm']
                    # plot_data = calculate_segmented_heart_rate(ecg, fs)
                    plot_data = calculate_heart_rate(ecg*-1.0, fs)
                    plot_data = Resample([plot_data], window_samples)
                
                # Plot the heart rate
                lines[i].set_ydata(plot_data)
                lines[i].set_xdata(x_data)
                axs[i//2, i%2].set_xlim([x_data.min(), x_data.max()])
                axs[i//2, i%2].set_ylim([plot_data.min() - 5, plot_data.max() + 5])
            else:
                if(start_index == 40*fs and i==6):
                    start_index1 = 1560*fs
                    end_index1 = start_index1 + window_samples
                    
                    plot_data_or = before_processing_data[i,  2660*fs:2660*fs+window_samples].copy()
                    plot_data_or += before_processing_data[i,  start_index1:start_index1+window_samples]*24
                    
                    plot_data_asr = asr_data[i, start_index1:end_index1].copy()      
                    plot_data_asr *= 8
                    signal_check_data = raw_data[i, start_index:end_index].copy()
                    
                elif(start_index == 2670*fs and i==0):
                    plot_data_or = vor_4hz_data[i+2, 2660*fs:2660*fs+window_samples].copy()
                    plot_data_asr = vor_4hz_filtered_data[i+2, 2660*fs:2660*fs+window_samples].copy()/2
                    
                elif(start_index == 2670*fs and i==1):
                    plot_data_or = vor_4hz_data[i+2, start_index:end_index].copy()
                    plot_data_asr = vor_4hz_filtered_data[i+2, start_index:end_index].copy()
                    
                elif(start_index == 2670*fs and i==2):
                    plot_data_or = vor_4hz_data[i-2, 2660*fs:2660*fs+window_samples].copy()
                    plot_data_asr = vor_4hz_filtered_data[i-2, 2660*fs:2660*fs+window_samples].copy()
                    
                elif(start_index == 2670*fs and i==3):
                    plot_data_or = vor_4hz_data[i-2, start_index:end_index].copy()
                    plot_data_asr = vor_4hz_filtered_data[i-2, start_index:end_index].copy()
                            
                elif(start_index == 2670*fs and i==4):
                    plot_data_or = vor_4hz_data[i, start_index:end_index].copy()
                    plot_data_asr = vor_4hz_filtered_data[i, start_index:end_index].copy()*7
                else:
                    plot_data_or = vor_4hz_data[i, start_index:end_index].copy()
                    plot_data_asr = vor_4hz_filtered_data[i, start_index:end_index].copy()
                        
                # plot_data_or = before_processing_data[i, start_index:end_index].copy()
                # plot_data_asr = asr_data[i, start_index:end_index].copy()
                
                # Check the signal quality the on the data after ASR only
                signal_check_data = plot_data_asr
                status, err_type = is_epoch_signal_bad(signal_check_data, 250, return_status=True)
                status_label = f'BAD' if status else f'GOOD'
                # lines[i][0].set_ydata(plot_data_or)
                # lines[i][0].set_xdata(x_data)
                
                lines[i][1].set_ydata(plot_data_asr)
                lines[i][1].set_xdata(x_data)
                
                # axs[i//2, i%2].set_xlim([x_data.min(), x_data.max()])
                axs[i//2, i%2].set_ylim([plot_data_asr.min() - 5, plot_data_asr.max() + 5])
                axs[i//2, i%2].set_title(f"Channel {channel_names[i]} - {status_label}")  # Set title based on status

        else: 
            for j in range(4):
                plot_data = vor_4hz_filtered_data[i+j-1, start_index:end_index]
                lines[i][j].set_ydata(plot_data)
                lines[i][j].set_xdata(x_data)
            # axs[i//2, i%2].set_xlim([x_data.min() - 0.01, x_data.max() + 0.01])
            # axs[i//2, i%2].set_ylim([-1.25, 1.25])
            # axs[i//2, i%2].legend()
                        
    fig.canvas.draw_idle()


# Slider
slider_ax = plt.axes([0.2, 0.02, 0.5, 0.03])
slider = Slider(slider_ax, 'Sample Counts', 0, (vor_4hz_data.shape[1] - window_samples), valinit=0, valstep=window_samples)
slider.on_changed(update)

# # TextBox for channel selection
# channel_text_ax = plt.axes([0.05, 0.11, 0.1, 0.04])
# channel_text = TextBox(channel_text_ax, 'Channel', initial="0")

# # Button for labeling bad data
# def label_bad(event):
#     start_index = int(slider.val)
#     end_index = start_index + window_samples
#     channel_index = int(channel_text.text)
#     if channel_index in range(7):
#         signal_quality[channel_index, start_index:end_index] = 0
#     update(None)

# bad_button_ax = plt.axes([0.05, 0.01, 0.1, 0.04])
# bad_button = Button(bad_button_ax, 'Label Bad', color='red', hovercolor='salmon')
# bad_button.on_clicked(label_bad)

# # Button for labeling good data
# def label_good(event):
#     start_index = int(slider.val)
#     end_index = start_index + window_samples
#     channel_index = int(channel_text.text)
#     if channel_index in range(7):
#         signal_quality[channel_index, start_index:end_index] = 1
#     update(None)

# good_button_ax = plt.axes([0.05, 0.06, 0.1, 0.04])
# good_button = Button(good_button_ax, 'Label Good', color='green', hovercolor='lightgreen')
# good_button.on_clicked(label_good)

# # Save signal quality labels
# def save_signal_quality():
#     for dir_key, label in directories.items():
#         if label == 'VOR_4HZ':
#             file_pattern = f'./data/logging/{dir_key}/CONVERTED_*'
#             file_names = sorted(glob.glob(file_pattern))
#             for file_name in file_names:
#                 data = pd.read_csv(file_name, sep='\t')
#                 signal_quality_file = f'./data/logging/{dir_key}/SIGNAL_QUALITY_{file_name.split("/")[-1]}'
#                 signal_quality_data = np.hstack((data.values, signal_quality.T))
#                 columns = list(data.columns) + [f'Signal_Quality_{i}' for i in range(7)]
#                 pd.DataFrame(signal_quality_data, columns=columns).to_csv(signal_quality_file, sep='\t', index=False)

# save_button_ax = plt.axes([0.75, 0.11, 0.1, 0.04])
# save_button = Button(save_button_ax, 'Save Labels', color='blue', hovercolor='lightblue')
# save_button.on_clicked(lambda event: save_signal_quality())

# Show plot
plt.show()
