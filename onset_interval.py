from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.stats import entropy
from ruptures import detection
import ruptures as rpt
from hmmlearn import hmm

def load_model_and_predict(channel, features, model_folder='ML_Models'):
    # Check for NaN values in the features
    if features.isna().any().any():
        print(f"NaN detected in channel: {channel}")
        print("Columns with NaN values:")
        print(features.columns[features.isna().any()].tolist())
        print("Rows with NaN values:")
        print(features[features.isna().any(axis=1)])
        raise ValueError(f"NaN values found in the data for channel: {channel}.")
    
    model_path = f'{model_folder}/{channel}_final_model_base_post_v3.joblib'
    model = joblib.load(model_path)
    
    return model.predict_proba(features)[:, 1]  

def sliding_window_entropy(probabilities, window_size=10, bins=2):
    entropy_values = []

    for i in range(len(probabilities) - window_size + 1):
        window = probabilities[i:i + window_size]  # Current window of probabilities
        hist, _ = np.histogram(window, bins=bins, range=(0, 1), density=False)
        hist = hist / np.sum(hist) if np.sum(hist) != 0 else hist
        window_entropy = entropy(hist)
        entropy_values.append(window_entropy)

    entropy_values = np.array(entropy_values)
    padded_entropy = np.concatenate([np.zeros(window_size - 1), entropy_values])
    
    return padded_entropy

def detect_onset_based_on_max_entropy(entropy_values):
    onset_index = np.argmax(entropy_values)
    return onset_index

def detect_change_points_bayesian(probabilities):
    model = "rbf"  # Radial Basis Function model for change point detection
    algo = rpt.Binseg(model=model).fit(probabilities)
    change_points = algo.predict(pen=10)  # The penalty parameter is automatically handled by the model.
    return change_points

def detect_onset_hmm(probabilities, n_states=2):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
    model.fit(probabilities.reshape(-1, 1))
    hidden_states = model.predict(probabilities.reshape(-1, 1))
    
    change_points = np.where(np.diff(hidden_states) != 0)[0] + 1  # Indices where the state changes
    return change_points

dir_names = [
    '1712094934_Nhan', 
    '1713985211_Mallory', 
    '1713999445_Wil', 
    '1713293584_Jacob', 
    '1713550894_AlexKagoda', 
    '1713820773_Andrew', 
    '1713994149_Saad', 
    '1713458084_Andrei', 
    '1714511822_Nathan'
]

stim = {'1712094934_Nhan': {'START1': 152682, 'ONSET1': None, 'END1': 185144, 'START2': 337670, 'ONSET2': 353901+30*250, 'END2': 370132, 'START3': 522856, 'ONSET3': 539087+20*250, 'END3': 555318, 'START4': 858098, 'ONSET4': 874329, 'END4': 890560}, '1713985211_Mallory': {'START1': 152519, 'ONSET1': 155041, 'END1': 184947, 'START2': 337496, 'ONSET2': 341941, 'END2': 370003, 'START3': 521934, 'ONSET3': 527943, 'END3': 542732, 'START4': 701929, 'ONSET4': 703008, 'END4': 733549}, '1713999445_Wil': {'START1': 152704, 'ONSET1': None, 'END1': 185166, 'START2': 337858, 'ONSET2': 350239, 'END2': 370200, 'START3': 522909, 'ONSET3': 531924, 'END3': 555391, 'START4': 725204, 'ONSET4': 729048, 'END4': 757546}, '1713293584_Jacob': {'START1': 152521, 'ONSET1': 155644, 'END1': 184983, 'START2': 337581, 'ONSET2': 339141, 'END2': 369910, 'START3': 522521, 'ONSET3': 527568, 'END3': 555009, 'START4': 740316, 'ONSET4': 741996, 'END4': 772658}, '1713550894_AlexKagoda': {'START1': 152680, 'ONSET1': None, 'END1': 185142, 'START2': 337726, 'ONSET2': 362372, 'END2': 370130, 'START3': 522701, 'ONSET3': 534883, 'END3': 555204, 'START4': 741344, 'ONSET4': 766952, 'END4': 773685}, '1713820773_Andrew': {'START1': 152635, 'ONSET1': None, 'END1': 185097, 'START2': 337767, 'ONSET2': None, 'END2': 370075, 'START3': 522695, 'ONSET3': 535677, 'END3': 555036, 'START4': 706607, 'ONSET4': 715261, 'END4': 739087}, '1713994149_Saad': {'START1': 152584, 'ONSET1': 156068, 'END1': 185037, 'START2': 304073, 'ONSET2': 317056, 'END2': 336501, 'START3': 489085, 'ONSET3': 509282, 'END3': 521547, 'START4': 651886, 'ONSET4': 668117, 'END4': 684348}, '1713458084_Andrei': {'START1': 167500, 'ONSET1': 236174, 'END1': 237500, 'START2': 420000, 'ONSET2': 432172, 'END2': 460000, 'START3': 627500, 'ONSET3': 633382, 'END3': 660500, 'START4': 848500, 'ONSET4': 853409, 'END4': 880000}, '1714511822_Nathan': {'START1': 152704, 'ONSET1': 164364, 'END1': 185166, 'START2': 316708, 'ONSET2': 326685, 'END2': 349046, 'START3': 423713, 'ONSET3': 428880, 'END3': 456175, 'START4': 539620, 'ONSET4': 552362, 'END4': 572082}}
drop_columns = ['Index', 'C4_hjorthMobility', 'C4_hjorthComplexity', 'EDA-Skew', 'EDA-Kurt']

combined_X = pd.DataFrame()
combined_metadata = pd.DataFrame() 
new_classes = {
    '2 levels': {
        'order': {0: 0, 1: 1}, 
        'Class Metrics': {str(i): {'Precision': [], 'Recall': [], 'F1-Score': []} for i in range(1, 3)}
    }
}

for dir_name in dir_names:
    df = pd.read_csv(f'data/logging/{dir_name}/whole_dataset.csv')
    X_temp = df.drop(drop_columns, axis=1)
    
    metadata_temp = pd.DataFrame({
        'Dataset': [dir_name] * len(X_temp),
        'Original_Index': range(len(X_temp))
    })
    
    combined_X = pd.concat([combined_X, X_temp], axis=0, ignore_index=True)
    combined_metadata = pd.concat([combined_metadata, metadata_temp], axis=0, ignore_index=True)

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(combined_X)
combined_X = pd.DataFrame(X_normalized, columns=combined_X.columns)
combined_metadata = combined_metadata.reset_index(drop=True)

channels = {
    'O1': [col for col in combined_X.columns if 'O1' in col],
    'O2': [col for col in combined_X.columns if 'O2' in col],
    'C3': [col for col in combined_X.columns if 'C3' in col],
    'C4': [col for col in combined_X.columns if 'C4' in col],
    'EOG': [col for col in combined_X.columns if 'EOG' in col],
    'ECG': ['HR', 'HRV'],
    'EDA': [col for col in combined_X.columns if 'EDA' in col]
}

# Group the features by channels
grouped_features = {channel: combined_X[features] for channel, features in channels.items()}
final_probabilities = np.zeros(len(combined_X))

for channel, features in grouped_features.items():
    print(f"Processing channel: {channel}")
    channel_probs = load_model_and_predict(channel, features)
    final_probabilities += channel_probs

final_probabilities /= len(grouped_features)

for dir_name in dir_names:
    # Create a new figure with 4 subplots arranged in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()  # Flatten to easily index the subplots

    # Get the dataset indices and smoothed average probabilities for the current subject
    dataset_indices = combined_metadata[combined_metadata['Dataset'] == dir_name].index
    avg_probs = final_probabilities[dataset_indices]
    smoothed_avg_probs = pd.Series(avg_probs).ewm(span=10).mean().to_numpy()

    for i in range(1, 5):
        start_key = f'START{i}'
        end_key = f'END{i}'
        
        start_idx = int(stim[dir_name][start_key] / 250)
        end_idx = int(stim[dir_name][end_key] / 250)
        
        # Entropy calculation for each stimulation section
        section_probs = smoothed_avg_probs[start_idx:end_idx]
        # entropy_values = sliding_window_entropy(section_probs, window_size=10)
        # dizziness_onset = detect_onset_based_on_max_entropy(entropy_values)
        
        # change_points = detect_change_points_bayesian(section_probs)
        change_points = detect_onset_hmm(section_probs)

        # Select the subplot
        ax = axs[i - 1]
        ax.plot(section_probs, label='Probability')
        # ax.plot(entropy_values, label='Entropy', color='orange')
        
        for cp in change_points:
            ax.axvline(x=cp, color='red', linestyle='--', label='Change Point')

        onset_key = f'ONSET{i}'
        if stim[dir_name][onset_key] is not None:
            onset_idx = int(stim[dir_name][onset_key] / 250) - start_idx
            ax.axvline(x=onset_idx, color='k', linestyle='--', label=f'Feedback')
        
        # Mark the dizziness onset if detected
        # if dizziness_onset is not None:
        #     ax.axvline(x=dizziness_onset, color='red', linestyle='--', label='Entropy Onset')

        # Customize the subplot
        ax.set_title(f"Section {i} - {dir_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Values")
        ax.legend()

    plt.suptitle(f"Average Probability and Entropy for {dir_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()


# for dir_name in dir_names:
#     plt.figure(figsize=(12, 8))
    
#     dataset_indices = combined_metadata[combined_metadata['Dataset'] == dir_name].index
#     avg_probs = final_probabilities[dataset_indices]
#     avg_probs = np.clip(avg_probs, 0, 1) * 100
    
#     avg_probs_series = pd.Series(avg_probs)
#     smoothed_avg_probs = avg_probs_series.ewm(span=10).mean()

    
#     plt.plot(smoothed_avg_probs, linewidth=1, label=f'Average Probability Across Channels')
    
#     plt.axvline(x=stim[dir_name]['START1']/250, color='red', linestyle='--')
#     plt.axvline(x=stim[dir_name]['END1']/250, color='red', linestyle='--')
#     plt.axvline(x=stim[dir_name]['START2']/250, color='red', linestyle='--')
#     plt.axvline(x=stim[dir_name]['END2']/250, color='red', linestyle='--')
#     plt.axvline(x=stim[dir_name]['START3']/250, color='red', linestyle='--')
#     plt.axvline(x=stim[dir_name]['END3']/250, color='red', linestyle='--')
#     plt.axvline(x=stim[dir_name]['START4']/250, color='red', linestyle='--')
#     plt.axvline(x=stim[dir_name]['END4']/250, color='red', linestyle='--')  
    
#     # Optional: add onset index line if you have it
#     # plt.axvline(x=int(onset_idx[dir_name]/fs), color='k', linestyle=':', label='Onset', linewidth=2)
#     # plt.text(int(onset_idx[dir_name]/fs), 1.05, 'Onset', color='k', ha='center', va='bottom', fontsize=10)
    
#     plt.title(f"Average Probability of Belonging to Class 1 for Dataset: {dir_name}")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Average Probability (%)")
#     plt.legend()
#     plt.tight_layout()
