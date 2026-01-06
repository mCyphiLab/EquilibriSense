import numpy as np
import matplotlib.pyplot as plt
from utils.signal_processing import filter_sos, filter_iir
from utils.heart_rate import HeartRateMonitorECG
from utils.channel_selection import is_epoch_signal_bad
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
class VisualizeDataOnline(object):  
    """A class for visualizing EEG, ECG, IMU, and EDA data."""
    # Down sample to reduce the plot data
    EEG_SAMPLING_RATE = 250  # Hz
    IMU_SAMPLING_RATE = 250  # Hz
    EDA_SAMPLING_RATE = 1  # Hz
    MICROVOLTS_PER_COUNT = (4500000) / 24 / (2**23 - 1)  # uV/count
    WINDOW_LEN_EEG = 1250
    WINDOW_LEN_IMU = 1250
    WINDOW_LEN_HR = 5
    WINDOW_LEN_EDA = 25 # since the sampling rate of EDA is very low
    LABELS = {'EEG':{
                'O1':[0,0,0], # 0 stares the raw data index, 1 and 2 are used for plot indices
                'O2':[1,0,1], 
                'C3':[2,1,0], 
                'C4':[3,1,1], 
                'EOG-H':[4,2,0], 
                'EOG-V':[5,2,1], 
                'ECG':[6,3,0]}, 
            'EDA':[0,3,1], 
            'IMU':{
                'Acc X':[0,4,1], 
                'Acc Y':[1,4,1], 
                'Acc Z':[2,4,1]},
                # Re-use ECG for Heart Rate   
            'Heart Rate':[6,4,0]}
    EEG_CHANNELS = 7
    IMU_CHANNELS = 3

    def __init__(self, plot_type):
        """Initialize the VisualizeData object with a specific plot type."""
        self.fig, self.axs = self._setup_figure(plot_type)
        self.legacy_data_eeg = [[0] * self.WINDOW_LEN_EEG for _ in range(self.EEG_CHANNELS)]
        self.legacy_data_imu = [[0] * self.WINDOW_LEN_IMU for _ in range(self.IMU_CHANNELS)]
        self.legacy_data_hr = np.zeros(self.WINDOW_LEN_HR)
        self.legacy_data_eda = np.zeros(self.WINDOW_LEN_EDA)
        self.times_eeg = np.arange(self.WINDOW_LEN_EEG) / self.EEG_SAMPLING_RATE
        self.times_imu = np.arange(self.WINDOW_LEN_IMU) / self.IMU_SAMPLING_RATE
        self.times_eda = np.arange(self.WINDOW_LEN_EDA) / self.EDA_SAMPLING_RATE
        self.times_hr = np.arange(self.WINDOW_LEN_HR)
        self.hr = HeartRateMonitorECG(self.EEG_SAMPLING_RATE)

    def _setup_figure(self, plot_type):
        """Set up the figure layout based on the plot type."""
        if plot_type == 'time':
            return plt.subplots(5, 2, figsize=(17, 7), sharex=False)
        elif plot_type == 'frequency':
            return plt.subplots(2, 3, figsize=(18, 10), sharex=False)
        else:
            raise ValueError("Invalid plot type specified.")

    def plot_time_data(self, data):
        """Plot time domain data for EEG, ECG, IMU, and EDA signals."""
        # Down sample from 250Hz to 125Hz
        downsampled_data = data[1:12, :]
        # downsampled_data = data[1:12, ::2]

        self._plot_eeg_channels(downsampled_data[:7,:])
        self._plot_imu_data(downsampled_data[8:11,:])
        self._plot_heart_rate()
        self._plot_eda_data(downsampled_data[7])
        
        plt.xticks(rotation=45, ha='right')
        self.fig.align_ylabels(self.axs)
        plt.draw()
        plt.pause(0.05) 

    def _plot_eeg_channels(self, eeg_data, spectrum=False):
        """Plot EEG channel data."""
        remove = eeg_data.shape[1]
        keep = self.WINDOW_LEN_EEG - remove
        state = None
        
        for channel in self.LABELS['EEG']:
            idx, x, y = self.LABELS['EEG'][channel][:]
            self.legacy_data_eeg[idx][:keep] = self.legacy_data_eeg[idx][remove:]
            self.legacy_data_eeg[idx][keep:] = eeg_data[idx]
            plot_data = np.array(self.legacy_data_eeg[idx]).copy()

            # TODO: Move this filtering to utils.signal_processing.py
            # DataFilter.detrend(plot_data, DetrendOperations.CONSTANT.value)
            # DataFilter.perform_bandpass(plot_data, self.EEG_SAMPLING_RATE, 1, 40.0, 4,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_bandstop(plot_data, self.EEG_SAMPLING_RATE, 48.0, 52.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_bandstop(plot_data, self.EEG_SAMPLING_RATE, 58.0, 62.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            
            # Apply bandpass filter
            # plot_data, state = filter_sos(plot_data, 'bandpass', [1, 40.0], self.EEG_SAMPLING_RATE, 4, zi=state)
            # plot_data, z2 = filter_sos(plot_data, 'notch', 50.0, self.EEG_SAMPLING_RATE, 2, notchWidth=4)
            # plot_data, z3 = filter_sos(plot_data, 'notch', 60.0, self.EEG_SAMPLING_RATE, 2, notchWidth=4)

            plot_data = filter_iir(plot_data, 'bandpass', [1, 50.0], self.EEG_SAMPLING_RATE, 4, zi=state)
            plot_data = filter_iir(plot_data, 'notch', 50.0, self.EEG_SAMPLING_RATE, 2, notch_width=4)
            plot_data = filter_iir(plot_data, 'notch', 60.0, self.EEG_SAMPLING_RATE, 2, notch_width=4)

            # Signal check
            status = is_epoch_signal_bad(plot_data, self.EEG_SAMPLING_RATE)
            status_disp = 'BAD' if status else 'GOOD'

            if spectrum:
                fft_data = np.fft.rfft(plot_data)
                plot_data = np.abs(fft_data)/len(fft_data)
                self._plot(self.axs[x, y], None, plot_data, labels=[f'{channel} - {status_disp}', 'Amplitude', 'Frequency (Hz)'], clear=True)    
            else:                
                self._plot(self.axs[x, y], self.times_eeg, plot_data, labels=[f'{channel} - {status_disp}', 'uV', 'Time (s)'], clear=True)    
 
    def _plot_imu_data(self, imu_data, spectrum=False):
        """Plot IMU sensor data."""
        remove = imu_data.shape[1]
        keep = self.WINDOW_LEN_IMU - remove
        clear = True
        
        for channel in self.LABELS['IMU']:
            idx, x, y = self.LABELS['IMU'][channel][:]
            self.legacy_data_imu[idx][:keep] = self.legacy_data_imu[idx][remove:] 
            self.legacy_data_imu[idx][keep:] = imu_data[idx, :]
            plot_data = self.legacy_data_imu[idx].copy()
            
            if spectrum:
                fft_data = np.fft.rfft(plot_data)
                plot_data = np.abs(fft_data)/len(fft_data)                
                self._plot(self.axs[x, y], None, plot_data, labels=[channel, 'Amplitude', 'Frequency (Hz)'], clear=clear)    
            else:   
                self._plot(self.axs[x, y], self.times_imu, plot_data, labels=[channel, 'Accel [g]', 'Time (s)'], clear=clear)    
            if clear:
                clear = False

    def _plot_eda_data(self, eda_data):
        """Plot Heart rate."""
        remove = 1
        keep = self.WINDOW_LEN_EDA - remove
        _, x, y = self.LABELS['EDA']
        self.legacy_data_eda[:keep] = self.legacy_data_eda[remove:]
        self.legacy_data_eda[keep:] = eda_data[:1]
        self._plot(self.axs[x, y], self.times_eda, self.legacy_data_eda, labels=['EDA', 'V', 'Time (s)'], clear=True)    
    
    def _plot_heart_rate(self):
        """Plot Heart rate."""
        idx, x, y = self.LABELS['Heart Rate'][:]
        ecg = np.array(self.legacy_data_eeg[idx]).copy()
        filtered_data = self.hr.filter_ecg(ecg)
        hr = self.hr.calculate_heart_rate(filtered_data, self.WINDOW_LEN_HR)
        self._plot(self.axs[x, y], self.times_hr, hr, labels=['Heart Rate', 'BPM', 'Time (s)'], clear=True)

    def _plot(self, 
              axs, # axs for the plot
              ax, # data on x axis
              ay, # data on y axis
              labels=[ # default babels for data
                  'data', # label for the data
                  'x', # label for x axis
                  'y' # label for y axis
              ],
              clear=False # clear previous display
              ):
        if clear:
            axs.clear()
        if ax is not None:
            axs.plot(ax, ay, label=labels[0])
        else:
            axs.plot(ay,  label=labels[0])

        axs.legend(fontsize=10, loc='upper right')
        axs.set_xlabel(labels[2], fontsize=12)
        axs.set_ylabel(labels[1], fontsize=12)
        
    def clear_plot(self):
        plt.clf()