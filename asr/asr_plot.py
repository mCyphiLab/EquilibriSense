import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np
from numpy.fft import fft, fftfreq
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from matplotlib.widgets import Button
from clean_artifacts import clean_artifacts_realtime
import scipy
import threading
import time

lock = threading.Lock()

############################################################################################
############################################################################################
############################################################################################
# Channel names and labels
channel_names = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz', 'accel']
channel_labels = ['uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'g (1g = 9.8m/s^2)']

file_names = sorted(glob.glob('./data/logging/Head_Turn_Testing/CONVERTED_*'))
data = pd.concat([pd.read_csv(file_name, sep='\t') for file_name in file_names])
data = data.drop(columns=['index', 'EDA', 'time stamp', 'red led', 'ir led', 'green', 'EEG - Pz', 'temperature', 'EDA'])

# Specific case for filling the IMU data with last valid value
columns_to_process = ['accel X', 'accel Y', 'accel Z']
# Loop over the columns and replace zeros with NaN, then forward fill the NaN values
for column in columns_to_process:
    data[column] = data[column].replace(0, np.nan)    
    data[column].ffill(inplace=True)    
    data[column].fillna(0, inplace=True)
        

# Downsampling data from 250Hz to 125Hz
data = data[::2]

sampling_rate = 125
total_samples = len(data)
window_samples = 5 * sampling_rate 

fig, axs = plt.subplots(8, 2, figsize=(15, 20), sharex="col")

lines = []
x_init = np.linspace(0, window_samples/sampling_rate, window_samples)
for i, channel_name in enumerate(channel_names):
    if i == 7:
        lines.append([axs[i, 0].plot(x_init, np.zeros(window_samples), label=axis, linewidth=0.5)[0] for axis in ['accel X', 'accel Y', 'accel Z']])
    else:
        lines.append(axs[i, 0].plot(x_init, np.zeros(window_samples), label=channel_name, linewidth=0.5)[0])
    axs[i, 0].set_title(channel_name)
    axs[i, 0].set_ylabel(channel_labels[i])

axs[-1, 0].set_xlabel('Time (seconds) from start of recording')

asr_ax = plt.axes([0.7, 0.01, 0.1, 0.05])
btn_asr = Button(asr_ax, 'Apply ASR')
apply_asr = False

# def asr_callback(event):
#     global apply_asr
#     print("ASR Button Pressed!")
#     if apply_asr == True:
#         apply_asr = False
#     else:
#         apply_asr = True

def asr_callback(event):
    global apply_asr
    with lock:
        apply_asr = not apply_asr
        print("ASR Button Pressed! New state:", apply_asr)

btn_asr.on_clicked(asr_callback)


def compute_psd(channel_data):
    yf = fft(channel_data)
    xf = fftfreq(len(channel_data), 1 / sampling_rate)
    psd = np.abs(yf) ** 2

    desired_freq_mask = (xf >= 0) & (xf <= 30)  # Keep only positive frequencies and up to 30Hz
    return xf[desired_freq_mask], psd[desired_freq_mask]


semaphore = threading.Semaphore(0)
shared_results = {}
frame_counter = 0  # Global frame counter

def worker_thread():
    global data, apply_asr, frame_counter

    button_toggle = False
    while True:
        semaphore.acquire()  # This will block until the semaphore is released
        
        with lock:  # Safely get the frame count
            frame = frame_counter

        start_index = frame * 25  
        end_index = start_index + window_samples

        if end_index > total_samples:
            return
        
        if apply_asr:

            filtered_data_list = []  # List to hold filtered data for each channel
            
            for i in range(7):
                line_data = data.iloc[start_index:end_index, i].to_numpy().copy()
                DataFilter.detrend(line_data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(line_data, 125, 5, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(line_data, 125, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
                
                filtered_data_list.append(line_data)
            
            # Concatenate the filtered data to have shape of (7, len(line_data))
            concatenated_data = np.vstack(filtered_data_list)
            
            # Assign the concatenated data to input_signal['data']
            input_signal = {
                'data': concatenated_data,
                'srate': 125,
                'nbchan': 7,
                'etc': {
                },
            }
            
            # Apply ASR
            output_signal = clean_artifacts_realtime(input_signal)['data']
            button_toggle = True
            # print(output_signal['data'].shape)


        for i in range(len(lines)):
            x_data = np.linspace(start_index/sampling_rate, end_index/sampling_rate, end_index-start_index) 
            if i == 7:  # accel channel
                for j in range(3):
                    line_data = data.iloc[start_index:end_index, i + j].to_numpy().copy()
                    DataFilter.detrend(line_data, DetrendOperations.CONSTANT.value)
                    DataFilter.perform_bandpass(line_data, 125, 0.5, 20.0, 2, FilterTypes.BUTTERWORTH.value, 0)
                    lines[i][j].set_ydata(line_data)
                    lines[i][j].set_xdata(x_data)
                    
                    if j==0:
                        # Compute PSD and plot
                        xf, psd = compute_psd(line_data)
                        axs[i, 1].clear()
                        axs[i, 1].plot(xf, psd, color='b')
                        axs[i, 1].set_title('PSD of ' + channel_names[i])
                        axs[i, 1].set_xlabel('Frequency (Hz)')
                        axs[i, 1].set_ylabel('PSD')
            else:            
                # Apply filters to EEG data
                if apply_asr and button_toggle:
                    button_toggle = False
                    line_data = output_signal[i,:]
                else:
                    line_data = data.iloc[start_index:end_index, i].to_numpy().copy()
                    DataFilter.detrend(line_data, DetrendOperations.CONSTANT.value)
                    DataFilter.perform_bandpass(line_data, 125, 0.5, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)
                    DataFilter.perform_bandstop(line_data, 125, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
                    
                lines[i].set_ydata(line_data)
                lines[i].set_xdata(x_data)
            
                # Compute PSD and plot
                xf, psd = compute_psd(line_data)
                axs[i, 1].clear()
                axs[i, 1].plot(xf, psd, color='b')
                axs[i, 1].set_title('PSD of ' + channel_names[i])
                axs[i, 1].set_xlabel('Frequency (Hz)')
                axs[i, 1].set_ylabel('PSD')
            
            axs[i, 0].relim()
            axs[i, 0].autoscale_view(True, True, True)
            axs[i, 1].relim()
            axs[i, 1].autoscale_view(True, True, True)
                
                # fig.canvas.draw_idle()

thread = threading.Thread(target=worker_thread)
thread.daemon = True  # This will ensure the thread exits when the main program exits
thread.start()


def update(frame):
    global frame_counter

    with lock:  # Safely update the frame count
        frame_counter = frame

    semaphore.release()  # Release the semaphore to signal the worker thread

ani = animation.FuncAnimation(fig, update, frames=(total_samples - window_samples) // 25, repeat=False, interval=200)
plt.tight_layout()
plt.show()
