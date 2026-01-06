
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.signal_processing import *
from utils.utils import *
from utils.eeglib.helpers import CSVCustomHelper, CSVHelper
from utils.eeglib.helpers import Helper
import re
import neurokit2 as nk
import time
from collections import Counter
import os

def find_onset_idx(name, subjective_feedback, chunks):
    subjective_feedback = np.array(subjective_feedback)
    onset = []
    for j in range(len(chunks)):
        start, stop = chunks[j]
        if person == '1712094934_Nhan' or person == '1713994149_Saad':
            onset_idx = start + (stop-start) // 2
        else:
            onset_idx = None
            
        for i in range(start, stop):
            if (subjective_feedback[i] >= 2) and (subjective_feedback[i-1] <= 1):
                
                print(f'Find onset: Rotation {j+1}')
                
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

stim = {
    '1712094934_Nhan': {}, 
    '1713985211_Mallory': {}, 
    '1713999445_Wil': {}, 
    '1713293584_Jacob': {}, 
    '1713550894_AlexKagoda': {},
    '1713820773_Andrew': {}, 
    '1713994149_Saad': {},
    '1713458084_Andrei': {}, 
    '1714511822_Nathan': {}
}


fs = 250
eeg_chans = 4


for person, directories in all_directories.items():
    print(f"Processing directories for {person}:")
    csvHelper= CSVCustomHelper(directories)
    
    ###################################################################################
    ###################################################################################
    raw_data, channel_names, length = csvHelper.data, csvHelper.names, csvHelper.lengths
    eeg_data, eog_data, ecg_data, eda_data, imu_data, subjective_feedback = raw_data[:4], raw_data[4:6], raw_data[6], raw_data[7], raw_data[8:11], raw_data[11]
    n_samples = raw_data.shape[1]    
    
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
        start_rest_stim1_idx =  start_stim1_idx + length['VOR_1HZ']
        
        start_stim2_idx = start_rest_stim1_idx + length['VOR_1HZ_REST']
        start_rest_stim2_idx =  start_stim2_idx + length['VOR_2HZ']
        
        start_stim3_idx = start_rest_stim2_idx + length['VOR_2HZ_REST']
        start_rest_stim3_idx =  start_stim3_idx + length['VOR_3HZ']
        
        start_stim4_idx = start_rest_stim3_idx + length['VOR_3HZ_REST']
        start_rest_stim4_idx =  start_stim4_idx + length['VOR_4HZ']
    
    ###################################################################################
    ###################################################################################
    onset_idx = find_onset_idx(person, subjective_feedback, [[start_stim1_idx, start_rest_stim1_idx], 
                                                     [start_stim2_idx, start_rest_stim2_idx],
                                                     [start_stim3_idx, start_rest_stim3_idx],
                                                     [start_stim4_idx, start_rest_stim4_idx]])
    
    stim[person]['START1'] = start_stim1_idx
    stim[person]['ONSET1'] = onset_idx[0]
    stim[person]['END1'] = start_rest_stim1_idx
    
    stim[person]['START2'] = start_stim2_idx
    stim[person]['ONSET2'] = onset_idx[1]
    stim[person]['END2'] = start_rest_stim2_idx
    
    stim[person]['START3'] = start_stim3_idx
    stim[person]['ONSET3'] = onset_idx[2]
    stim[person]['END3'] = start_rest_stim3_idx
    
    stim[person]['START4'] = start_stim4_idx
    stim[person]['ONSET4'] = onset_idx[3]
    stim[person]['END4'] = start_rest_stim4_idx

    print(stim)
    continue

    ###################################################################################
    ###################################################################################
    # Window sliding
    window_size_seconds = 10
    window_size_samples = window_size_seconds*fs
    
    vestibular_features = {}
    
    _window_stride_size_seconds = 1
    _window_stride_size_samples = _window_stride_size_seconds*fs
    
    ###################################################################################
    ###################################################################################
    # Labeling
    # for _onset_idx in onset_idx:
    #     adjusted_subjective_feedback = subjective_feedback.copy()
    #     adjusted_subjective_feedback[:] = 0.0    
    #     adjusted_subjective_feedback[onset_idx:] = 1.

    #     vestibular_features['Labels'] = label_segments(adjusted_subjective_feedback, window_size_samples, _window_stride_size_samples)
    
    ###################################################################################
    ###################################################################################
    dataframes = []

    # Process each feature array to reshape and convert to DataFrame
    for feature_name, array in vestibular_features.items():
        feature_df = pd.DataFrame(array, columns=[feature_name])
        dataframes.append(feature_df)

    # Create an index DataFrame
    index_df = pd.DataFrame(np.arange(len(vestibular_features['HR'])), columns=['Index'])

    # Concatenate all feature dataframes alongside the index
    final_df = pd.concat([index_df] + dataframes, axis=1)

    # Specify the file path
    file_path = f'data/logging/{person}/whole_dataset1.csv'

    # Check if file exists
    if os.path.exists(file_path):
        # Delete the file if it exists
        os.remove(file_path)

    # Save the final DataFrame to CSV
    final_df.to_csv(file_path, index=False)