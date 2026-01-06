import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
from utils.signal_processing import filter_iir, filter_sos
from scipy.integrate import cumulative_trapezoid
import neurokit2 as nk
import mne
from scipy.stats import kurtosis, skew


def float_to_byte_string(float_value):
    # convert to int
    int_value = int(float_value)

    # handle two's complement
    if int_value < 0:
        int_value = int_value + 2**16

    # convert to hex
    hex_value = hex(int_value)[2:]
    if len(hex_value) % 2 != 0:  # Ensure that the hex string has even length by padding with a leading zero if necessary
        hex_value = '0' + hex_value

    hex_value = hex_value.upper().encode('utf-8')

    # convert to byte string
    try:
        byte_string = bytes.fromhex(hex_value.decode('utf-8'))
    except ValueError as e:
        print(f"Could not convert '{hex_value}' to byte string. Error: {e}")
        raise

    return byte_string

def reverse_scale_raw(scaled_value, channel_gain):
    VOLTAGE_REF = 4.5  # Voltage reference
    ADC_MAX_VALUE = pow(2, 23) - 1  # Maximum value for a 24-bit ADC
    MICROVOLT_CONVERSION = 1000000.0  # Conversion factor to microvolts

    eeg_scale = (VOLTAGE_REF / ADC_MAX_VALUE) / channel_gain * MICROVOLT_CONVERSION
    original_value = scaled_value / eeg_scale
    return original_value

def reverse_scale(scaled_value, channel_gain, expected_gain):
    VOLTAGE_REF = 4.5  # Voltage reference
    ADC_MAX_VALUE = pow(2, 23) - 1  # Maximum value for a 24-bit ADC
    MICROVOLT_CONVERSION = 1000000.0  # Conversion factor to microvolts

    eeg_scale = (VOLTAGE_REF / ADC_MAX_VALUE) / channel_gain * MICROVOLT_CONVERSION
    expected_eeg_scale = (VOLTAGE_REF / ADC_MAX_VALUE) / expected_gain * MICROVOLT_CONVERSION
    desired_value = (scaled_value * expected_eeg_scale) / eeg_scale
    return desired_value

def detect_motion(sum_square, fs, windows=10, segment=2, segment_overlap=1, threshold=[2e-4, 5e-4]):
    motion_indices = []
    
    window_samples = windows*fs
    segment_samples = segment*fs
    segment_overlap_samples = segment_overlap*fs
    low_threshold, high_threshold = threshold
    len_samples = len(sum_square)

    for window_start in range(0, len_samples, window_samples):
        window_end = window_start + window_samples  
        if window_end > len_samples:
            break
        
        segment_indices = []

        for segment_start in range(window_start, window_end, segment_samples - segment_overlap_samples):
            segment_end = segment_start + segment_samples
            motion_detected = 0

            if segment_end > len_samples:
                break 

            segment = sum_square[segment_start:segment_end]
            average_energy = compute_average_energy(np.array(segment))

            if average_energy >= high_threshold:
                # Definitely motion detection
                motion_detected = 1
            
            elif average_energy >= low_threshold and average_energy < high_threshold:
                # Uncertain motion detection
                motion_detected = 2
                
            segment_indices.append(motion_detected)

        if all(x == 0 for x in segment_indices):
            motion_label = 0
        elif 1 in segment_indices:
            motion_label = 1
        else:
            motion_label = 2
            
        motion_indices.append([motion_label, segment_indices])
        
        # if motion_detected:
        #     window_indice = int(window_start/window_samples)
            
    return motion_indices

def compute_vector_sum(accel_x_filtered=None, accel_y_filtered=None, accel_z_filtered=None):
    return np.sqrt(accel_x_filtered**2 + accel_y_filtered**2 + accel_z_filtered**2).tolist()

def min_max_scaling_to_eeg(imu_data, eeg_data):
    eeg_min = np.min(eeg_data)
    eeg_max = np.max(eeg_data)
    imu_min = np.min(imu_data)
    imu_max = np.max(imu_data)
    scaled_imu_data = (imu_data - imu_min) / (imu_max - imu_min) * (eeg_max - eeg_min) + eeg_min
    return scaled_imu_data

def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    normalized_signal = (signal - mean) / std
    return normalized_signal, (mean, std)

def convert_back_to_original_scale(processed_signal, params):
    mean, std = params
    original_scale_signal = processed_signal * std + mean
    return original_scale_signal

def universal_threshold(signal):
    N = len(signal)
    sigma = estimate_sigma(signal, average_sigmas=True)
    thresh = sigma * np.sqrt(2 * np.log(N))
    return thresh

def compute_average_energy(signal):
    average_energy = np.mean(signal**2)
    return average_energy

def adjust_imu_feature(imu_feature, lag_value):
    # Your function as provided
    if lag_value > 0:
        adjusted_feature = np.concatenate((np.zeros(lag_value), imu_feature))[:-lag_value]
    elif lag_value < 0:
        adjusted_feature = np.concatenate((imu_feature[abs(lag_value):], np.zeros(abs(lag_value))))
    else:
        adjusted_feature = imu_feature
    return adjusted_feature

def plot_normalized(data, labels, axes_labels, title):
    plt.figure(figsize=(15, 4))    
    for i in range(len(data)):
        plot_data, _ = z_score_normalize(data[i])
        plt.plot(plot_data, linewidth=0.5, label=labels[i])
        
    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    plt.title(title)
    plt.legend()
    
def extract_imu_feature(data, fs, bandpass=[1, 50]):
    # data shape (N channels, M samples)
    accel = []

    for i in range(len(data)):
        filtered = filter_iir(data[i], 'bandpass', bandpass, fs, 4)
        # filtered,_ = filter_sos(filtered_data[i], 'bandpass', [1, 50], fs, 4)
        accel.append(filtered)
        
    sum_square_ma = compute_vector_sum(accel[0], accel[1], accel[2])

    dt = 1/fs
    velocity = [cumulative_trapezoid(y=a, dx=dt, initial=0) for a in accel]
    velocity = np.array(velocity)
    sum_square_velocity = compute_vector_sum(velocity[0], velocity[1], velocity[2])

    imu_features = {
        'Vel-X':velocity[0], 
        'Vel-Y':velocity[1], 
        'Vel-Z':velocity[2], 
        'Accel-X':accel[0], 
        'Accel-Y':accel[1], 
        'Accel-Z':accel[2], 
        'Vel-Square':sum_square_velocity, 
        'Accel-Square':sum_square_ma
        }
    
    return imu_features

def extract_subjective_feedback(feedback):
    feedback_changes = []
    last_feedback = None

    for i in range(len(feedback)):
        current_feedback = feedback[i]
        if current_feedback != last_feedback:
            feedback_changes.append([i, current_feedback])
            last_feedback = current_feedback 

    return feedback_changes

def extract_blink_rate(signal, fs, windows, windows_overlap):
    eog_names = ['horizontal', 'vertical']
    eog_info = mne.create_info(ch_names=eog_names, sfreq=fs, ch_types=['eog', 'eog'])

    _eog = signal.copy()
    raw = mne.io.RawArray(_eog, eog_info)

    all_blink_events = []
    for start in range(0, _eog.shape[1], windows_overlap):
        end = start + windows
        if end > _eog.shape[1]:
            break

        raw_segment = raw.copy().crop(tmin=start/fs, tmax=(end-1)/fs)    
        blink_events = mne.preprocessing.find_eog_events(raw_segment, filter_length=1250, verbose=False, ch_name='vertical')
        all_blink_events.append(blink_events)

    # all_blink_events = np.concatenate(all_blink_events, axis=0)
    return all_blink_events


def extract_blink_rate_custom(signal, fs, windows, window_stride):
    eog_names = ['horizontal', 'vertical']
    eog_info = mne.create_info(ch_names=eog_names, sfreq=fs, ch_types=['eog', 'eog'])

    _eog = signal.copy()
    raw = mne.io.RawArray(_eog, eog_info)

    all_blink_events = []
    
    window_start = 0
    while window_start < _eog.shape[1]:        
        window_stride_size_samples = window_stride

        window_end = window_start + windows
        if window_end > _eog.shape[1]:
            break

        raw_segment = raw.copy().crop(tmin=window_start/fs, tmax=(window_end-1)/fs)    
        blink_events = mne.preprocessing.find_eog_events(raw_segment, filter_length=1250, verbose=False, ch_name='vertical')
        all_blink_events.append(blink_events)
        
        window_start += window_stride_size_samples

    # all_blink_events = np.concatenate(all_blink_events, axis=0)
    return all_blink_events

def extract_nystagmus_freq(signal, fs, windows):
    # Placeholder for nystagmus frequency extraction logic
    # Return the frequency of nystagmus oscillations
    return np.random.random()  # Dummy implementation

def extract_nystagmus_amplitude(signal, fs, windows):
    # Placeholder for nystagmus amplitude extraction logic
    # Return the amplitude of nystagmus beats
    return np.max(np.abs(signal)) - np.min(np.abs(signal))

def extract_nystagmus_direction(signal, fs, windows):
    # Placeholder for nystagmus direction extraction logic
    # Return the general direction of nystagmus (e.g., left or right)
    return np.sign(np.mean(np.diff(signal)))

def extract_statistical_features(signal, windows, window_overlap_size_samples):
    mean_val = []
    var_val = []
    skew_val = []
    kurt_val = []

    for start in range(0, len(signal), window_overlap_size_samples):
        end = start + windows
        if end > len(signal):
            break
        segment = signal[start:end]
        mean_val.append(np.mean(segment))
        var_val.append(np.var(segment))
        skew_val.append(skew(segment))
        kurt_val.append(kurtosis(segment))
        
    return mean_val, var_val, skew_val, kurt_val

def extract_statistical_features_custom(signal, windows, window_stride):
    mean_val = []
    var_val = []
    skew_val = []
    kurt_val = []

    window_start = 0
    while window_start < len(signal):        
        window_stride_size_samples = window_stride

        window_end = window_start + windows
        if window_end > len(signal):
            break

        segment = signal[window_start:window_end]
        mean_val.append(np.mean(segment))
        var_val.append(np.var(segment))
        skew_val.append(skew(segment))
        kurt_val.append(kurtosis(segment))
        
        window_start += window_stride_size_samples
        
    return mean_val, var_val, skew_val, kurt_val
