import numpy as np
import scipy.signal as signal
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

class HeartRateMonitorPPG:
    def __init__(self):
        self.IR_AC_Max = 20
        self.IR_AC_Min = -20
        self.IR_AC_Signal_Current = 0
        self.IR_AC_Signal_Previous = 0
        self.IR_AC_Signal_min = 0
        self.IR_AC_Signal_max = 0
        self.IR_Average_Estimated = 0
        self.positiveEdge = 0
        self.negativeEdge = 0
        self.ir_avg_reg = 0
        self.cbuf = np.zeros(32)
        self.offset = 0
        self.FIRCoeffs = np.array([172, 321, 579, 927, 1360, 1858, 2390, 2916, 3391, 3768, 4012, 4096])

        self.last_beat = 0
        self.RATE_SIZE = 4
        self.rates = [0] * 4
        self.rateSpot = 0
        self.data_others_time = 0
        self.beatsPerMinute  = 0
        self.beatAvg = 0
        self.lastBeat = 0

    def getHeartrate(self, ir_value):
        # due to the sampling rate of 62.5 Hz
        if self.checkForBeat(ir_value):  # Assume we have a `check_for_beat` function
            # We sensed a beat!
            delta = self.data_others_time - self.lastBeat
            self.lastBeat = self.data_others_time

            self.beatsPerMinute = 60 / (delta / 1000.0)

            if 20 < self.beatsPerMinute < 255:
                self.rates[self.rateSpot] = self.beatsPerMinute  # Store this reading in the array
                self.rateSpot += 1
                self.rateSpot %= self.RATE_SIZE  # Wrap variable

                # Take average of readings
                self.beatAvg = int(sum(self.rates) / len(self.rates))
        
        self.data_others_time += 16

        return self.beatAvg

    def checkForBeat(self, sample):
        beatDetected = False
        self.IR_AC_Signal_Previous = self.IR_AC_Signal_Current
        self.IR_Average_Estimated = self.averageDCEstimator(sample)
        self.IR_AC_Signal_Current = self.lowPassFIRFilter(sample - self.IR_Average_Estimated)

        if ((self.IR_AC_Signal_Previous < 0) & (self.IR_AC_Signal_Current >= 0)):
            self.IR_AC_Max = self.IR_AC_Signal_max
            self.IR_AC_Min = self.IR_AC_Signal_min
            self.positiveEdge = 1
            self.negativeEdge = 0
            self.IR_AC_Signal_max = 0
            if (((self.IR_AC_Max - self.IR_AC_Min) > 20) & ((self.IR_AC_Max - self.IR_AC_Min) < 1000)):
                beatDetected = True

        if ((self.IR_AC_Signal_Previous > 0) & (self.IR_AC_Signal_Current <= 0)):
            self.positiveEdge = 0
            self.negativeEdge = 1
            self.IR_AC_Signal_min = 0

        if self.positiveEdge & (self.IR_AC_Signal_Current > self.IR_AC_Signal_Previous):
            self.IR_AC_Signal_max = self.IR_AC_Signal_Current

        if self.negativeEdge & (self.IR_AC_Signal_Current < self.IR_AC_Signal_Previous):
            self.IR_AC_Signal_min = self.IR_AC_Signal_Current

        return beatDetected

    def averageDCEstimator(self, x):
        self.ir_avg_reg = int(self.ir_avg_reg)  # convert to native Python int
        x = int(x)  # convert to native Python int
        self.ir_avg_reg  += ((x << 15) - self.ir_avg_reg ) // 4
        return self.ir_avg_reg // (1 << 15)

    def lowPassFIRFilter(self, din):
        self.cbuf[self.offset] = din
        z = self.FIRCoeffs[11] * self.cbuf[(self.offset - 11) & 0x1F]

        for i in range(11):
            z += self.FIRCoeffs[i] * (self.cbuf[(self.offset - i) & 0x1F] + self.cbuf[(self.offset - 22 + i) & 0x1F])

        self.offset += 1
        self.offset %= 32
        return z // (1 << 15)

    def mul16(self, x, y):
        return x * y


class HeartRateMonitorECG:
    def __init__(self, fs):
        self.fs = fs
    
    def filter_ecg(self, data):
        ecg = data
        # Filter the ecg signal first
        DataFilter.detrend(ecg, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(ecg, 250, 5, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(ecg, 250, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(ecg, 250, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)

        # DataFilter.detrend(ecg, DetrendOperations.CONSTANT.value)
        # b, a = signal.butter(4, 50/(self.fs/2), 'low')

        # tempf = signal.filtfilt(b,a, ecg)
        # ### Compute Kaiser window co-effs to eliminate baseline drift noise ###
        # nyq_rate = self.fs/ 2.0
        # # The desired width of the transition from pass to stop.
        # width = 5.0/nyq_rate
        # # The desired attenuation in the stop band, in dB.
        # ripple_db = 60.0
        # # Compute the order and Kaiser parameter for the FIR filter.
        # O, beta = signal.kaiserord(ripple_db, width)
        # # The cutoff frequency of the filter.
        # cutoff_hz = 4.0
        #                                                         ###Use firwin with a Kaiser window to create a lowpass FIR filter.###
        # taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
        # # Use lfilter to filter x with the FIR filter.
        # filtered_data = signal.lfilter(taps, 1.0, tempf)
        filtered_data = ecg
        
        return filtered_data
    
    def calculate_heart_rate(self, data, resample):
        threshold = np.max(data) * 0.5  # Simple threshold based on segment mean
        peaks, _ = signal.find_peaks(data, height=threshold, distance=self.fs/4)
        # Calculate RR intervals and heart rate if there are enough peaks
        if len(peaks) > 1:
            RR_intervals = np.diff(peaks) / self.fs
            average_RR_interval = np.mean(RR_intervals)
            heart_rate = 60 / average_RR_interval
        else:
            # heart_rates.append(None)  # Indicate that heart rate couldn't be calculated for this segment
            heart_rate = 0
        
        heart_rate = self.Resample([heart_rate], resample)
        return heart_rate
    
    def Resample(self, hb, samples):
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