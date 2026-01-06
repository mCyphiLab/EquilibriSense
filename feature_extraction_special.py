
import glob
import numpy as np
import pandas as pd
import scipy.fftpack    
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.signal_processing import *
from utils.channel_selection import is_epoch_signal_bad
from utils.utils import *
from utils.eeglib.helpers import CSVCustomHelper, CSVHelper
from utils.eeglib.helpers import Helper
import re
import neurokit2 as nk
import time
from collections import Counter
import os


def find_onset_idx(subjective_feedback, chunks):
    subjective_feedback = np.array(subjective_feedback)
    onset = []
    for start, stop in chunks:
        onset_idx = start + (stop-start) // 2
        for i in range(start, stop):
            if (subjective_feedback[i] >= 2) and (subjective_feedback[i-1] <= 1):
                onset_idx = i
                break
    
        onset.append(onset_idx)
    return onset

def label_segments(feedback, window_size_samples, window_stride_size_samples):
    n_samples = len(feedback)
    true_labels = []
    window_start = 0
    
    while window_start < n_samples:                
        window_end = window_start + window_size_samples
        if window_end > n_samples:
            break

        segment = feedback[window_start:window_end]
        
        counter = Counter(segment)
        count_0 = counter.get(0.0, 0)
        count_1 = counter.get(1.0, 0)
        
        if count_1 > count_0:
            true_labels.append(1.0)
        else:
            true_labels.append(0.0)
        
        window_start += window_stride_size_samples
        
    true_labels = np.array(true_labels)
    return true_labels


all_directories = {
    '1712094934_Nhan' : {
        '1712094934_Nhan/1712094934_BASE_LINE': 'BASE_LINE',
        '1712094934_Nhan/1712095682_VOR_1HZ': 'VOR_1HZ',
        '1712094934_Nhan/1712095831_VOR_1HZ_REST': 'VOR_1HZ_REST',
        '1712094934_Nhan/1712096475_VOR_2HZ': 'VOR_2HZ',
        '1712094934_Nhan/1712096619_VOR_2HZ_REST': 'VOR_2HZ_REST',
        '1712094934_Nhan/1712097275_VOR_3HZ': 'VOR_3HZ',
        '1712094934_Nhan/1712097418_VOR_3HZ_REST': 'VOR_3HZ_REST',
        '1712094934_Nhan/1712098656_VOR_4HZ': 'VOR_4HZ',
        '1712094934_Nhan/1712098799_VOR_4HZ_REST': 'VOR_4HZ_REST'
    },
    
    '1713985211_Mallory': {
        '1713985211_Mallory/1712270250_BASE_LINE': 'BASE_LINE',
        '1713985211_Mallory/1712270977_VOR_1HZ': 'VOR_1HZ',
        '1713985211_Mallory/1712271123_VOR_1HZ_REST': 'VOR_1HZ_REST',
        '1713985211_Mallory/1712271878_VOR_2HZ': 'VOR_2HZ',
        '1713985211_Mallory/1712272019_VOR_2HZ_REST': 'VOR_2HZ_REST',
        '1713985211_Mallory/1712272675_VOR_3HZ': 'VOR_3HZ',
        '1713985211_Mallory/1712272768_VOR_3HZ_REST': 'VOR_3HZ_REST',
        '1713985211_Mallory/1712273445_VOR_4HZ': 'VOR_4HZ',
        '1713985211_Mallory/1712273580_VOR_4HZ_REST': 'VOR_4HZ_REST'
    },

    '1713999445_Wil': {
        '1713999445_Wil/1712262127_BASE_LINE': 'BASE_LINE',
        '1713999445_Wil/1712262819_VOR_1HZ': 'VOR_1HZ',
        '1713999445_Wil/1712263003_VOR_1HZ_REST': 'VOR_1HZ_REST',
        '1713999445_Wil/1712263704_VOR_2HZ': 'VOR_2HZ',
        '1713999445_Wil/1712263852_VOR_2HZ_REST': 'VOR_2HZ_REST',
        '1713999445_Wil/1712264509_VOR_3HZ': 'VOR_3HZ',
        '1713999445_Wil/1712264650_VOR_3HZ_REST': 'VOR_3HZ_REST',
        '1713999445_Wil/1712265369_VOR_4HZ': 'VOR_4HZ',
        '1713999445_Wil/1712265511_VOR_4HZ_REST': 'VOR_4HZ_REST'
    },

    '1713293584_Jacob': {
        '1713293584_Jacob/1713293584_BASE_LINE': 'BASE_LINE',
        '1713293584_Jacob/1713294328_VOR_1HZ': 'VOR_1HZ',
        '1713293584_Jacob/1713294472_VOR_1HZ_REST': 'VOR_1HZ_REST',
        '1713293584_Jacob/1713295133_VOR_2HZ': 'VOR_2HZ',
        '1713293584_Jacob/1713295276_VOR_2HZ_REST': 'VOR_2HZ_REST',
        '1713293584_Jacob/1713295919_VOR_3HZ': 'VOR_3HZ',
        '1713293584_Jacob/1713296059_VOR_3HZ_REST': 'VOR_3HZ_REST',
        '1713293584_Jacob/1713296831_VOR_4HZ': 'VOR_4HZ',
        '1713293584_Jacob/1713296971_VOR_4HZ_REST': 'VOR_4HZ_REST'
    },

    '1713550894_AlexKagoda' : {
        '1713550894_AlexKagoda/1713550894_BASE_LINE': 'BASE_LINE',
        '1713550894_AlexKagoda/1713551593_VOR_1HZ': 'VOR_1HZ',
        '1713550894_AlexKagoda/1713551737_VOR_1HZ_REST': 'VOR_1HZ_REST',
        '1713550894_AlexKagoda/1713552395_VOR_2HZ': 'VOR_2HZ',
        '1713550894_AlexKagoda/1713552534_VOR_2HZ_REST': 'VOR_2HZ_REST',
        '1713550894_AlexKagoda/1713553173_VOR_3HZ': 'VOR_3HZ',
        '1713550894_AlexKagoda/1713553317_VOR_3HZ_REST': 'VOR_3HZ_REST',
        '1713550894_AlexKagoda/1713554089_VOR_4HZ': 'VOR_4HZ',
        '1713550894_AlexKagoda/1713554229_VOR_4HZ_REST': 'VOR_4HZ_REST'
    },

    '1713820773_Andrew': {
        '1713820773_Andrew/1713820773_BASE_LINE': 'BASE_LINE',
        '1713820773_Andrew/1713821448_VOR_1HZ': 'VOR_1HZ',
        '1713820773_Andrew/1713821595_VOR_1HZ_REST': 'VOR_1HZ_REST',
        '1713820773_Andrew/1713822396_VOR_2HZ': 'VOR_2HZ',
        '1713820773_Andrew/1713822537_VOR_2HZ_REST': 'VOR_2HZ_REST',
        '1713820773_Andrew/1713823177_VOR_3HZ': 'VOR_3HZ',
        '1713820773_Andrew/1713823316_VOR_3HZ_REST': 'VOR_3HZ_REST',
        '1713820773_Andrew/1713823947_VOR_4HZ': 'VOR_4HZ',
        '1713820773_Andrew/1713824087_VOR_4HZ_REST': 'VOR_4HZ_REST'
    },

    # Saad
    '1713994149_Saad': {
        '1713994149_Saad/1713994149_BASE_LINE': 'BASE_LINE',
        '1713994149_Saad/1713994824_VOR_1HZ': 'VOR_1HZ',
        '1713994149_Saad/1713994964_VOR_1HZ_REST': 'VOR_1HZ_REST',
        '1713994149_Saad/1713995471_VOR_2HZ': 'VOR_2HZ',
        '1713994149_Saad/1713995610_VOR_2HZ_REST': 'VOR_2HZ_REST',
        '1713994149_Saad/1713996271_VOR_3HZ': 'VOR_3HZ',
        '1713994149_Saad/1713996412_VOR_3HZ_REST': 'VOR_3HZ_REST',
        '1713994149_Saad/1713996958_VOR_4HZ': 'VOR_4HZ',
        '1713994149_Saad/1713997104_VOR_4HZ_REST': 'VOR_4HZ_REST'
    }, 
    
    '1713458084_Andrei': {
        '1713458084_Andrei/1713458084_BASE_LINE': 'BASE_LINE',
    },

    # Nathan
    '1714511822_Nathan': {
        '1714511822_Nathan/1714511822_BASE_LINE': 'BASE_LINE',
        '1714511822_Nathan/1714512555_VOR_1HZ': 'VOR_1HZ',
        '1714511822_Nathan/1714512696_VOR_1HZ_REST': 'VOR_1HZ_REST',
        '1714511822_Nathan/1714513251_VOR_2HZ': 'VOR_2HZ',
        '1714511822_Nathan/1714513389_VOR_2HZ_REST': 'VOR_2HZ_REST',
        '1714511822_Nathan/1714513724_VOR_3HZ': 'VOR_3HZ',
        '1714511822_Nathan/1714513865_VOR_3HZ_REST': 'VOR_3HZ_REST',
        '1714511822_Nathan/1714514222_VOR_4HZ': 'VOR_4HZ',
        '1714511822_Nathan/1714514362_VOR_4HZ_REST': 'VOR_4HZ_REST'
    }, 
}

# onset = {
#     '1712094934_Nhan': {}, 
#     '1713985211_Mallory': {}, 
#     '1713999445_Wil': {}, 
#     '1713293584_Jacob': {}, 
#     '1713550894_AlexKagoda': {},
#     '1713820773_Andrew': {}, 
#     '1713994149_Saad': {},
#     '1713458084_Andrei': {}, 
#     '1714511822_Nathan': {}
# }


fs = 250
eeg_chans = 4

for person, directories in all_directories.items():
    print(f"Processing directories for {person}:")
    csvHelper= CSVCustomHelper(directories)
    
    ###################################################################################
    ###################################################################################
    raw_data, channel_names, length = csvHelper.data, csvHelper.names, csvHelper.lengths    
    after_stim = 0

    ###################################################################################
    ###################################################################################
    start_base_idx = 0

    if person == '1713458084_Andrei':                
        start_stim1_idx = 670*fs
        start_rest_stim1_idx = 950*fs
        
        start_stim2_idx = 1680*fs
        start_rest_stim2_idx = 1840*fs
        
        start_stim3_idx = 2510*fs
        start_rest_stim3_idx = 2642*fs
        
        start_stim4_idx = 3394*fs
        start_rest_stim4_idx = 3520*fs
    else:
        # Dizziness onset        
        start_stim1_idx = length['BASE_LINE']
        start_rest_stim1_idx = start_stim1_idx + length['VOR_1HZ']
        
        start_stim2_idx = start_rest_stim1_idx + length['VOR_1HZ_REST']
        start_rest_stim2_idx = start_stim2_idx + length['VOR_2HZ']
        
        start_stim3_idx = start_rest_stim2_idx + length['VOR_2HZ_REST']
        start_rest_stim3_idx = start_stim3_idx + length['VOR_3HZ']
        
        start_stim4_idx = start_rest_stim3_idx + length['VOR_3HZ_REST']
        start_rest_stim4_idx = start_stim4_idx + length['VOR_4HZ']
    
    
    # Remove the redundant data during the rest post-stimulation (excluding the first after_stim seconds)
    raw_data = np.concatenate((raw_data[:, :start_rest_stim1_idx + after_stim*fs], 
                               raw_data[:,start_stim2_idx:start_rest_stim2_idx + after_stim*fs],
                               raw_data[:,start_stim3_idx:start_rest_stim3_idx + after_stim*fs],
                               raw_data[:,start_stim4_idx:start_rest_stim4_idx + 60*fs],
                               ), axis=1)

    ######1#############################################################################
    ###################################################################################
    # Remain the same for start and stop stim1, update from stim 2 to stim 4
    # start_stim1_idx
    # start_rest_stim1_idx
    post_stim1_idx = start_rest_stim1_idx + after_stim*fs
    
    if person == '1713458084_Andrei':   
        start_stim2_idx = post_stim1_idx
        start_rest_stim2_idx = start_stim2_idx + (1840-1680)*fs
        post_stim2_idx = start_rest_stim2_idx + after_stim*fs

        start_stim3_idx = post_stim2_idx
        start_rest_stim3_idx = start_stim3_idx + (2642-2510)*fs
        post_stim3_idx = start_rest_stim3_idx + after_stim*fs
        
        start_stim4_idx = post_stim3_idx
        start_rest_stim4_idx = start_stim4_idx + (3520-3394)*fs
        post_stim4_idx = start_rest_stim4_idx + 60*fs  
    else:
        start_stim2_idx = post_stim1_idx
        start_rest_stim2_idx = start_stim2_idx + length['VOR_2HZ']
        post_stim2_idx = start_rest_stim2_idx + after_stim*fs

        start_stim3_idx = post_stim2_idx
        start_rest_stim3_idx = start_stim3_idx + length['VOR_3HZ']
        post_stim3_idx = start_rest_stim3_idx + after_stim*fs
        
        start_stim4_idx = post_stim3_idx
        start_rest_stim4_idx = start_stim4_idx + length['VOR_2HZ']
        post_stim4_idx = start_rest_stim4_idx + 60*fs  

    ######1#############################################################################
    ###################################################################################
    eeg_data, eog_data, ecg_data, eda_data, imu_data, subjective_feedback = raw_data[:4], raw_data[4:6], raw_data[6], raw_data[7], raw_data[8:11], raw_data[11]
    n_samples = raw_data.shape[1]    
    
    ###################################################################################
    ###################################################################################
    onset_idx = find_onset_idx(subjective_feedback, [[start_stim1_idx, start_rest_stim1_idx], 
                                                     [start_stim2_idx, start_rest_stim2_idx],
                                                     [start_stim3_idx, start_rest_stim3_idx],
                                                     [start_stim4_idx, start_rest_stim4_idx]])
    
    stim_idx = 0
        
    filtered_eeg = filter_eeg(eeg_data.copy(), fs, [5,50])
    filtered_eog = filter_eeg(eog_data.copy(), fs, [1,50])
    filtered_ecg = filter_ecg(ecg_data.copy(), fs, [5,50], sos=False)
    imu_features = extract_imu_feature(imu_data.copy(), fs)
        
    ###################################################################################
    ###################################################################################
    # Window sliding
    window_size_seconds = 10
    window_size_samples = window_size_seconds*fs
    
    vestibular_features = {}
    
    _window_stride_size_seconds = 1
    _window_stride_size_samples = _window_stride_size_seconds*fs
        
        
    print('EEG processing')
    eegHelper = [Helper(filtered_eeg[i:i+1], 
        sampleRate=fs, 
        windowSize=window_size_samples, 
        names=channel_names[i], 
        normalize=False, 
        ICA=True,
        selectedSignals=False) for i in range(eeg_chans)]
    
    # Extract EEG features
    eeg_absoluteBandPower = []
    eeg_relativeBandPower = []
    eeg_bandPowerRatio = []
    eeg_pfd = []
    eeg_hfd = []
    eeg_hjorthActivity = []
    eeg_hjorthMobility = []
    eeg_hjorthComplexity = []
    eeg_sampEn = []
    eeg_LZC = []

    for i in range(eeg_chans):
        absoluteBandPower = []
        relativeBandPower = []
        bandPowerRatio = []
        pfd = []
        hfd = []
        hjorthActivity = []
        hjorthMobility = []
        hjorthComplexity = []
        sampEn = []
        LZC = []

        window_start = 0
        while window_start < n_samples:            
            window_stride_size_samples = _window_stride_size_samples

            window_end = window_start + window_size_samples
            
            if window_end > n_samples:
                break

            # Slide the window
            eegHelper[i].moveEEGWindow(window_start)
            
            # Extract the EEG features
            # segment_psd.append(eegHelper.eeg.PSD())
            absoluteBandPower.append(eegHelper[i].eeg.bandPower())
            relativeBandPower.append(eegHelper[i].eeg.bandPower(normalize=True))

            pfd.append(eegHelper[i].eeg.PFD())
            hfd.append(eegHelper[i].eeg.HFD())
            hjorthActivity.append(eegHelper[i].eeg.hjorthActivity())
            hjorthMobility.append(eegHelper[i].eeg.hjorthMobility())
            hjorthComplexity.append(eegHelper[i].eeg.hjorthComplexity())
            sampEn.append(eegHelper[i].eeg.sampEn())
            LZC.append(eegHelper[i].eeg.LZC())
            
            window_start += window_stride_size_samples
        
        # Append for each channel
        eeg_absoluteBandPower.append(absoluteBandPower)
        eeg_relativeBandPower.append(relativeBandPower)
        eeg_pfd.append(pfd)
        eeg_hfd.append(hfd)
        eeg_hjorthActivity.append(hjorthActivity)
        eeg_hjorthMobility.append(hjorthMobility)
        eeg_hjorthComplexity.append(hjorthComplexity)
        eeg_sampEn.append(sampEn)
        eeg_LZC.append(LZC)
        
    eeg_absoluteBandPower = np.array(eeg_absoluteBandPower).squeeze(axis=-1)
    eeg_relativeBandPower = np.array(eeg_relativeBandPower).squeeze(axis=-1)
    eeg_pfd = np.array(eeg_pfd).squeeze(axis=-1)
    eeg_hfd = np.array(eeg_hfd).squeeze(axis=-1)
    eeg_hjorthActivity = np.array(eeg_hjorthActivity).squeeze(axis=-1)
    eeg_hjorthMobility = np.array(eeg_hjorthMobility).squeeze(axis=-1)
    eeg_hjorthComplexity = np.array(eeg_hjorthComplexity).squeeze(axis=-1)
    eeg_sampEn = np.array(eeg_sampEn).squeeze(axis=-1)
    eeg_LZC = np.array(eeg_LZC).squeeze(axis=-1)
    
    eeg_absoluteBandPowerTheta = np.array([[freq_dict['theta'] for freq_dict in channel] for channel in eeg_absoluteBandPower])
    eeg_absoluteBandPowerAlpha = np.array([[freq_dict['alpha'] for freq_dict in channel] for channel in eeg_absoluteBandPower])
    eeg_absoluteBandPowerBeta = np.array([[freq_dict['beta'] for freq_dict in channel] for channel in eeg_absoluteBandPower])
    eeg_absoluteBandPowerGamma = np.array([[freq_dict['gamma'] for freq_dict in channel] for channel in eeg_absoluteBandPower])
    absolutePower = np.array([eeg_absoluteBandPowerTheta.copy(), eeg_absoluteBandPowerAlpha.copy(), eeg_absoluteBandPowerBeta.copy(), eeg_absoluteBandPowerGamma.copy()])

    eeg_relativeBandPowerTheta = np.array([[freq_dict['theta'] for freq_dict in channel] for channel in eeg_relativeBandPower])
    eeg_relativeBandPowerAlpha = np.array([[freq_dict['alpha'] for freq_dict in channel] for channel in eeg_relativeBandPower])
    eeg_relativeBandPowerBeta = np.array([[freq_dict['beta'] for freq_dict in channel] for channel in eeg_relativeBandPower])
    eeg_relativeBandPowerGamma = np.array([[freq_dict['gamma'] for freq_dict in channel] for channel in eeg_relativeBandPower])
    relativePower = np.array([eeg_relativeBandPowerTheta.copy(), eeg_relativeBandPowerAlpha.copy(), eeg_relativeBandPowerBeta.copy(), eeg_relativeBandPowerGamma.copy()])

    eegOtherFeatures = np.array([eeg_pfd.copy(), eeg_hfd.copy(), eeg_hjorthActivity.copy(), eeg_hjorthMobility.copy(), eeg_hjorthComplexity.copy(), eeg_sampEn.copy(), eeg_LZC.copy()])

    eeg_labels = {'O1':0, 'O2':1, 'C3':2, 'C4':3}

    features = {'absolute_power_theta':0, 'absolute_power_alpha':1, 'absolute_power_beta':2, 'absolute_power_gamma':3,
                'relative_power_theta':0, 'relative_power_alpha':1, 'relative_power_beta':2, 'relative_power_gamma':3,
                'power_ratio_theta_alpha':0, 'power_ratio_theta_beta':1, 'power_ratio_theta_gamma':2,
                'power_ratio_alpha_theta':0, 'power_ratio_alpha_beta':1, 'power_ratio_alpha_gamma':2,
                'power_ratio_beta_theta':0, 'power_ratio_beta_alpha':1, 'power_ratio_beta_gamma':2,
                'power_ratio_gamma_theta':0, 'power_ratio_gamma_alpha':1, 'power_ratio_gamma_beta':2,
                'PFD':0, 'HFD':1, 'hjorthActivity':2, 'hjorthMobility':3, 'hjorthComplexity':4, 'sampEn':5, 'LZC':6}

    for feature in features:
        for label in eeg_labels:
            key = f'{label}_{feature}'
            if 'absolute_power' in key:
                vestibular_features[key] = absolutePower[features[feature], eeg_labels[label]]
            elif 'relative_power' in key:
                vestibular_features[key] = relativePower[features[feature], eeg_labels[label]]
            elif 'power_ratio' in key:
                pattern = r'ratio_([a-zA-Z]+)_([a-zA-Z]+)'
                match = re.search(pattern, key)
                above, below = match.groups()
                power_above = absolutePower[features[f'absolute_power_{above}'], eeg_labels[label]]
                power_below = absolutePower[features[f'absolute_power_{below}'], eeg_labels[label]]
                vestibular_features[key] = power_above/power_below
            else:
                vestibular_features[key] = eegOtherFeatures[features[feature], eeg_labels[label]]
    
    
    ###################################################################################
    ###################################################################################
    # Extract ECG features
    print('ECG processing')
    hr = []
    hrv = []

    # Initialize placeholders for previous successful signals and info
    previous_signals, previous_info = None, None

    window_start = 0
    while window_start < n_samples:        
        window_stride_size_samples = _window_stride_size_samples
            
        window_end = window_start + window_size_samples
        if window_end > n_samples:
            break

        segment = filtered_ecg[window_start:window_end]

        try:
            signals, info = nk.ecg_process(segment, sampling_rate=fs, method='pantompkins1985')
            # Update previous_signals and previous_info on successful processing
            previous_signals, previous_info = signals, info
        except Exception as e:
            print(f"Error processing segment from {window_start} to {window_end}: {str(e)}")
            # Use the last successful signals and info if an error occurs
            if previous_signals is not None and previous_info is not None:
                signals, info = previous_signals, previous_info
            else:
                # If previous_signals is None, it means error occurred in the first segment
                continue  # Skip this loop iteration if no previous data to use

        # Calculate HR and HRV indices only if signals were successfully processed or retrieved
        if signals is not None:
            hr.append(np.mean(signals['ECG_Rate']))
            hrv_indices = nk.hrv_time(signals['ECG_R_Peaks'], sampling_rate=fs, show=False)
            hrv.append(hrv_indices['HRV_MeanNN'])
            
        window_start += window_stride_size_samples
        
    _series_hr = pd.Series(hr)
    smoothed_hr = _series_hr.ewm(span=30).mean()
    smoothed_hr = smoothed_hr.values

    _series_hrv = pd.Series(hrv)
    smoothed_hrv = _series_hrv.ewm(span=30).mean()
    smoothed_hrv = smoothed_hrv.values
    
    vestibular_features['HR'] = smoothed_hr
    vestibular_features['HRV'] = smoothed_hrv
    
    
    ###################################################################################
    ###################################################################################
    # Extract EOG
    print('EOG processing')
    all_blink_events = extract_blink_rate_custom(eog_data.copy(), fs, window_size_samples, _window_stride_size_samples)

    eog_h_mean_val, eog_h_var, eog_h_skew, eog_h_kurt = extract_statistical_features_custom(eog_data[0].copy(), 
                                                                                            window_size_samples, 
                                                                                            _window_stride_size_samples)
    
    eog_v_mean_val, eog_v_var, eog_v_skew, eog_v_kurt = extract_statistical_features_custom(eog_data[1].copy(), 
                                                                                            window_size_samples, 
                                                                                            _window_stride_size_samples)
    
    blink_events = np.concatenate(all_blink_events, axis=0)
    eog_blinks = np.array([len(events) for events in all_blink_events])
    eog_h_mean_val, eog_h_var, eog_h_skew, eog_h_kurt = np.array(eog_h_mean_val), np.array(eog_h_var), np.array(eog_h_skew), np.array(eog_h_kurt)
    eog_v_mean_val, eog_v_var, eog_v_skew, eog_v_kurt = np.array(eog_v_mean_val), np.array(eog_v_var), np.array(eog_v_skew), np.array(eog_v_kurt)
    
    vestibular_features['EOG-EyeBlink'] = eog_blinks

    vestibular_features['EOG-H-Mean'] = eog_h_mean_val
    vestibular_features['EOG-H-Var'] = eog_h_var
    vestibular_features['EOG-H-Skew'] = eog_h_skew
    vestibular_features['EOG-H-Kurt'] = eog_h_kurt

    vestibular_features['EOG-V-Mean'] = eog_v_mean_val
    vestibular_features['EOG-V-Var'] = eog_v_var
    vestibular_features['EOG-V-Skew'] = eog_v_skew
    vestibular_features['EOG-V-Kurt'] = eog_v_kurt
    
    
    ###################################################################################
    ###################################################################################
    # Extract EDA
    print('EDA processing')
    # Re-sample EDA to 5hz
    eda_fs = 5
    eda_mean_val, eda_var, eda_skew, eda_kurt = extract_statistical_features_custom(eda_data.copy(), 
                                                                                    window_size_samples,  
                                                                                    _window_stride_size_samples)
    eda_mean_val, eda_var, eda_skew, eda_kurt = np.array(eda_mean_val), np.array(eda_var), np.array(eda_skew), np.array(eda_kurt)

    
    vestibular_features['EDA-Mean'] = eda_mean_val
    vestibular_features['EDA-Var'] = eda_var
    vestibular_features['EDA-Skew'] = eda_skew
    vestibular_features['EDA-Kurt'] = eda_kurt
    
    ###################################################################################
    ###################################################################################
    # Labeling
    adjusted_subjective_feedback = subjective_feedback.copy()
    
    adjusted_subjective_feedback[:] = 0.0    
    adjusted_subjective_feedback[onset_idx[0]: post_stim1_idx] = 1.
    adjusted_subjective_feedback[onset_idx[1]: post_stim2_idx] = 1.
    adjusted_subjective_feedback[onset_idx[2]: post_stim3_idx] = 1.
    adjusted_subjective_feedback[onset_idx[3]: post_stim4_idx] = 1.

    vestibular_features['Labels'] = label_segments(adjusted_subjective_feedback, window_size_samples, _window_stride_size_samples)
    
    ###################################################################################
    ###################################################################################
    dataframes = []

    # Process each feature array to reshape and convert to DataFrame
    for feature_name, array in vestibular_features.items():
        feature_df = pd.DataFrame(array, columns=[feature_name])
        dataframes.append(feature_df)

    # Create an index DataFrame
    index_df = pd.DataFrame(np.arange(len(vestibular_features['Labels'])), columns=['Index'])

    # Concatenate all feature dataframes alongside the index
    final_df = pd.concat([index_df] + dataframes, axis=1)

    # Specify the file path
    file_path = f'data/logging/{person}/whole_dataset4.csv'

    # Check if file exists
    if os.path.exists(file_path):
        # Delete the file if it exists
        os.remove(file_path)

    # Save the final DataFrame to CSV
    final_df.to_csv(file_path, index=False)