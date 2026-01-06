import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from utils.signal_processing import filter_eeg, filter_ecg
from utils.utils import extract_imu_feature
from utils.eeglib.helpers import CSVCustomHelper, Helper
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def find_onset_idx(subjective_feedback):
    """
    Find the onset index where subjective feedback changes.
    """
    subjective_feedback = np.array(subjective_feedback)
    onset_idx = len(subjective_feedback) // 2
    
    for i in range(1, len(subjective_feedback)):
        if (subjective_feedback[i] >= 2) and (subjective_feedback[i-1] <= 1):
            onset_idx = i
            break
    
    return onset_idx


def process_directories(person, directories, fs=250, eeg_chans=4):
    """
    Process directories to extract EEG features.
    """
    print(f"Processing directories for {person}:")
    
    # Initialize CSVHelper based on the person
    if person == '1713458084_Andrei':
        stim_idx = 3394 * fs
        post_stim_idx = 3520 * fs
        csvHelper = CSVCustomHelper(directories, cutoff=[(0, stim_idx), (post_stim_idx, -1)])
    else:
        print('directories: ', directories)
        csvHelper = CSVCustomHelper(directories)
    
    # Extract raw data
    raw_data, channel_names, length = csvHelper.data, csvHelper.names, csvHelper.lengths
    eeg_data, eog_data, ecg_data, eda_data, imu_data, subjective_feedback = raw_data[:4], raw_data[4:6], raw_data[6], raw_data[7], raw_data[8:11], raw_data[11]
    n_samples = raw_data.shape[1]
    
    # Find onset index
    onset_idx = find_onset_idx(subjective_feedback)
    print(f'{person}: ', onset_idx)
    
    
    # Filter data
    filtered_eeg = filter_eeg(eeg_data.copy(), fs, [5, 50])
    filtered_eog = filter_eeg(eog_data.copy(), fs, [1, 50])
    filtered_ecg = filter_ecg(ecg_data.copy(), fs, [5, 50], sos=False)
    imu_features = extract_imu_feature(imu_data.copy(), fs)
    
    # Extract features using window sliding
    vestibular_features = extract_eeg_features(filtered_eeg, channel_names, fs, eeg_chans, n_samples)
    
    return filtered_eeg, vestibular_features, onset_idx


def extract_eeg_features(filtered_eeg, channel_names, fs, eeg_chans, n_samples, window_size_seconds=10, stride_seconds=1):
    """
    Extract EEG features using window sliding.
    """
    window_size_samples = window_size_seconds * fs
    stride_samples = stride_seconds * fs
    
    vestibular_features = {}
    eegHelpers = [Helper(filtered_eeg[i:i+1], sampleRate=fs, windowSize=window_size_samples, names=channel_names[i], normalize=False, ICA=True, selectedSignals=False) for i in range(eeg_chans)]
    
    features_containers = initialize_feature_containers(eeg_chans)
    
    for i in range(eeg_chans):
        window_start = 0
        while window_start < n_samples:
            window_end = window_start + window_size_samples
            if window_end > n_samples:
                break

            # Slide the window
            eegHelpers[i].moveEEGWindow(window_start)
            
            append_features(eegHelpers[i], features_containers, i)
            
            window_start += stride_samples
    
    features_containers = convert_to_numpy(features_containers)
    vestibular_features = organize_features(features_containers, eeg_chans)
    
    return vestibular_features


def initialize_feature_containers(eeg_chans):
    """
    Initialize containers for EEG features.
    """
    features_containers = {
        'absoluteBandPower': [[] for _ in range(eeg_chans)],
        'relativeBandPower': [[] for _ in range(eeg_chans)],
        'PFD': [[] for _ in range(eeg_chans)],
        'HFD': [[] for _ in range(eeg_chans)],
        'hjorthActivity': [[] for _ in range(eeg_chans)],
        'hjorthMobility': [[] for _ in range(eeg_chans)],
        'hjorthComplexity': [[] for _ in range(eeg_chans)],
        'sampEn': [[] for _ in range(eeg_chans)],
        'LZC': [[] for _ in range(eeg_chans)],
    }
    return features_containers


def append_features(eegHelper, features_containers, chan_idx):
    """
    Append features for a given channel.
    """
    features_containers['absoluteBandPower'][chan_idx].append(eegHelper.eeg.bandPower())
    features_containers['relativeBandPower'][chan_idx].append(eegHelper.eeg.bandPower(normalize=True))
    features_containers['PFD'][chan_idx].append(eegHelper.eeg.PFD())
    features_containers['HFD'][chan_idx].append(eegHelper.eeg.HFD())
    features_containers['hjorthActivity'][chan_idx].append(eegHelper.eeg.hjorthActivity())
    features_containers['hjorthMobility'][chan_idx].append(eegHelper.eeg.hjorthMobility())
    features_containers['hjorthComplexity'][chan_idx].append(eegHelper.eeg.hjorthComplexity())
    features_containers['sampEn'][chan_idx].append(eegHelper.eeg.sampEn())
    features_containers['LZC'][chan_idx].append(eegHelper.eeg.LZC())


def convert_to_numpy(features_containers):
    """
    Convert feature lists to numpy arrays and squeeze.
    """
    for key in features_containers:
        features_containers[key] = np.array(features_containers[key]).squeeze(axis=-1)
    return features_containers


def organize_features(features_containers, eeg_chans):
    """
    Organize features into a dictionary.
    """
    vestibular_features = {}
    
    eeg_labels = {'O1': 0, 'O2': 1, 'C3': 2, 'C4': 3}
    features = {
        'absolute': ['theta', 'alpha', 'beta', 'gamma'],
        'relative': ['theta', 'alpha', 'beta', 'gamma'],
        'power_ratio': ['theta_alpha', 'theta_beta', 'theta_gamma', 'alpha_theta', 'alpha_beta', 'alpha_gamma', 'beta_theta', 'beta_alpha', 'beta_gamma', 'gamma_theta', 'gamma_alpha', 'gamma_beta'],
        'PFD': 0, 'HFD': 1, 'hjorthActivity': 2, 'hjorthMobility': 3, 'hjorthComplexity': 4, 'sampEn': 5, 'LZC': 6
    }
    
    for feature_type, feature_keys in features.items():
        if feature_type in ['absolute', 'relative']:
            for band in feature_keys:
                for label in eeg_labels:
                    key = f'{label}_{feature_type}_{band}'
                    vestibular_features[key] = [bp[band] for bp in features_containers[f'{feature_type}BandPower'][eeg_labels[label]]]
        elif feature_type == 'power_ratio':
            for ratio in feature_keys:
                pattern = r'([a-zA-Z]+)_([a-zA-Z]+)'
                match = re.search(pattern, ratio)
                above, below = match.groups()
                for label in eeg_labels:
                    key = f'{label}_power_ratio_{ratio}'
                    power_above = np.array([bp[above] for bp in features_containers['absoluteBandPower'][eeg_labels[label]]])
                    power_below = np.array([bp[below] for bp in features_containers['absoluteBandPower'][eeg_labels[label]]])
                    vestibular_features[key] = power_above / power_below
        else:
            for label in eeg_labels:
                key = f'{label}_{feature_type}'
                vestibular_features[key] = features_containers[feature_type][eeg_labels[label]]
    
    return vestibular_features

def plot_stft(eeg_data, channel_names, onset_idx=None, fs=250, nperseg=250, max_freq=50):
    """
    Plot the Short-Time Fourier Transform (STFT) for each EEG channel using raw data.

    Args:
    - eeg_data: Raw EEG data for each channel.
    - channel_names: List of channel names.
    - onset_idx: Index indicating the onset of the event (optional).
    - fs: Sampling frequency (default: 250 Hz).
    - nperseg: Length of each segment for STFT (default: 250 samples).
    - max_freq: Maximum frequency to display (default: 50 Hz).
    """
    eeg_chans = len(channel_names)
    fig, axs = plt.subplots(eeg_chans, 1, figsize=(12, 15))
    fig.suptitle('Short-Time Fourier Transform (STFT) of Raw EEG Data')

    for i in range(eeg_chans):
        data = eeg_data[i]

        # Compute STFT
        f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg)

        # Limit the frequency to below max_freq (e.g., 50 Hz)
        freq_limit = f <= max_freq
        f = f[freq_limit]
        Sxx = Sxx[freq_limit, :]

        # Log scale and clip the spectrogram for better visualization
        Sxx = np.log1p(Sxx)
        Sxx = np.clip(Sxx, a_min=np.percentile(Sxx, 5), a_max=np.percentile(Sxx, 95))

        # Plot the spectrogram
        img = axs[i].pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
        axs[i].set_title(f'STFT - {channel_names[i]}')
        axs[i].set_ylabel('Frequency [Hz]')
        axs[i].set_xlabel('Time [s]')
        fig.colorbar(img, ax=axs[i])

        # Plot the onset index line
        if onset_idx is not None:
            axs[i].axvline(x=onset_idx/fs, color='r', linestyle='--', label='Onset')
            axs[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])




def visualize_clusters(combined_features_array, labels):
    """
    Visualize the clusters using the raw feature data.

    Args:
    - combined_features_array: The array of combined relative power features.
    - labels: Cluster labels for each feature calculation.
    """
    unique_labels = np.unique(labels)
    
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        cluster_points = combined_features_array[labels == label]
        plt.scatter(range(len(cluster_points)), cluster_points, label=f'Cluster {label}')
    
    plt.title('Cluster Visualization of Relative Power Features')
    plt.xlabel('Index')
    plt.ylabel('Relative Power Feature Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_clusters_tsne(scaled_features, labels):
    """
    Visualize the clusters using t-SNE.

    Args:
    - scaled_features: The scaled array of combined relative power features.
    - labels: Cluster labels for each feature calculation.
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(scaled_features)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_points = tsne_results[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
    
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()


def plot_cluster_time_distribution(labels, metadata, n_clusters):
    """
    Plot the distribution of time indices for each cluster as continuous colored bars,
    without specifying y-values, and remove gaps and black lines between classes.
    
    Args:
    - labels: Array of cluster labels for each time index.
    - metadata: List of tuples containing (subject, time_index) for each sample.
    - n_clusters: Number of clusters.
    """
    subjects = sorted(set([subject for subject, _ in metadata]))
    colors = plt.cm.get_cmap('tab10', n_clusters)  # Using tab10 colormap for distinct colors

    for subject in subjects:
        plt.figure(figsize=(12, 4))
        
        # Extract the relevant time indices and labels for this subject
        subject_indices = [i for i, (s, _) in enumerate(metadata) if s == subject]
        time_indices = [metadata[i][1] for i in subject_indices]
        subject_labels = [labels[i] for i in subject_indices]
        
        # Plot the cluster distribution as continuous colored bars
        for cluster_id in range(n_clusters):
            cluster_times = [time_indices[i] for i in range(len(time_indices)) if subject_labels[i] == cluster_id]
            plt.bar(cluster_times, [1]*len(cluster_times), color=colors(cluster_id), 
                    alpha=0.7, label=f'Cluster {cluster_id}', width=1.0)
        
        plt.title(f'Cluster Time Distribution for {subject}')
        plt.xlabel('Time (s)')
        plt.yticks([])  # Remove y-axis ticks since the y-values are no longer meaningful
        plt.legend(loc='upper right')


def plot_cluster_time_distribution_with_onset(labels, metadata, n_clusters, onset_dict=None):
    """
    Plot the distribution of time indices for each cluster as continuous colored bars,
    and add a dashed vertical line at the onset index for each subject.
    
    Args:
    - labels: Array of cluster labels for each time index.
    - metadata: List of tuples containing (subject, time_index) for each sample.
    - n_clusters: Number of clusters.
    - onset_dict: Dictionary with subjects as keys and onset indices as values (optional).
    """
    subjects = sorted(set([subject for subject, _ in metadata]))
    colors = plt.cm.get_cmap('tab10', n_clusters)  # Using tab10 colormap for distinct colors

    for subject in subjects:
        plt.figure(figsize=(12, 4))
        
        # Extract the relevant time indices and labels for this subject
        subject_indices = [i for i, (s, _) in enumerate(metadata) if s == subject]
        time_indices = [metadata[i][1] for i in subject_indices]
        subject_labels = [labels[i] for i in subject_indices]
        
        # Plot the cluster distribution as continuous colored bars
        for cluster_id in range(n_clusters):
            cluster_times = [time_indices[i] for i in range(len(time_indices)) if subject_labels[i] == cluster_id]
            plt.bar(cluster_times, [1]*len(cluster_times), color=colors(cluster_id), 
                    alpha=0.7, label=f'Cluster {cluster_id}', width=1.0)
        
        # Plot the onset index line if available for the current subject
        if onset_dict is not None and subject in onset_dict:
            onset_idx = onset_dict[subject]
            if onset_idx in time_indices:
                plt.axvline(x=onset_idx, color='k', linestyle=':', label='Onset', linewidth=2)
                # Add a text label indicating onset
                plt.text(onset_idx, 1.05, 'Onset', color='k', ha='center', va='bottom', fontsize=10)
        
        plt.title(f'Cluster Time Distribution for {subject}')
        plt.xlabel('Time (s)')
        plt.yticks([])  # Remove y-axis ticks since the y-values are no longer meaningful
        plt.ylim(0, 1.1)  # Set y-limits to ensure the text label fits
        plt.legend(loc='upper right')

if __name__ == "__main__":
    all_directories = {
        '1712094934_Nhan': {
            '1712094934_Nhan/1712098656_VOR_4HZ': 'VOR_4HZ',
        },
        
        '1712262127_Brian': {
            '1712262127_Brian/1712265369_VOR_4HZ': 'VOR_4HZ',
        },

        '1712270250_Onila': {
            '1712270250_Onila/1712273445_VOR_4HZ': 'VOR_4HZ',
        },

        '1713293584_Jacob': {
            '1713293584_Jacob/1713296831_VOR_4HZ': 'VOR_4HZ',
        },

        '1713550894_AlexKagoda' : {
            '1713550894_AlexKagoda/1713554089_VOR_4HZ': 'VOR_4HZ',
        },

        '1713820773_Andrew': {
            '1713820773_Andrew/1713823947_VOR_4HZ': 'VOR_4HZ',
        },

        # Saad
        '1713994149_Saad': {
            '1713994149_Saad/1713996958_VOR_4HZ': 'VOR_4HZ',
        }, 
        
        # Nathan
        '1714511822_Nathan': {
            '1714511822_Nathan/1714514222_VOR_4HZ': 'VOR_4HZ',
        }, 
    }

    fs = 250
    eeg_chans = 4
    
    features_list = []
    eeg_labels = ['O1', 'O2', 'C3', 'C4']
    all_vestibular_features = []
    onset = {}

    for person, directories in all_directories.items():
        eeg, vestibular_features, onset_idx = process_directories(person, directories, fs, eeg_chans)

        # plot_band_power_with_least_mean_square(vestibular_features, feature_type='relative', onset_idx=int(onset_idx/fs))
        # plot_stft(eeg, ['O1', 'O2', 'C3', 'C4'], onset_idx=onset_idx, fs=fs)
        # plt.show()

        all_vestibular_features.append((person, vestibular_features))
        onset[person] = int(onset_idx/fs)
        
        

