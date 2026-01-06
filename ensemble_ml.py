from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Helper function to extract average metrics
def extract_average_metrics(report):
    average_metrics = report['weighted avg']
    return average_metrics['precision'], average_metrics['recall'], average_metrics['f1-score']

# Helper function to map labels
def map_labels(y, classes):
    return y.map(classes)

# Initialize data and metadata storage
dir_names = [
    '1712094934_Nhan', '1713985211_Mallory', '1713999445_Wil', 
    '1713293584_Jacob', '1713550894_AlexKagoda', '1713820773_Andrew', 
    '1713994149_Saad', '1713458084_Andrei', '1714511822_Nathan'
]

# Drop columns that are not needed
drop_columns = ['Index', 'C4_hjorthMobility', 'C4_hjorthComplexity', 'EDA-Skew', 'EDA-Kurt']

# Define the initial empty DataFrame and classes
combined_X = pd.DataFrame()
combined_metadata = pd.DataFrame()  # Initialize metadata DataFrame
new_classes = {
    '2 levels': {
        'order': {0: 0, 1: 1},  # Correctly mapping only 2 classes
        'labels': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'Class Metrics': {str(i): {'Precision': [], 'Recall': [], 'F1-Score': []} for i in range(1, 3)}
    }
}

data_dict = {}

for dir_name in dir_names:
    df = pd.read_csv(f'data/logging/{dir_name}/whole_dataset4.csv')
    X_temp = df.drop(drop_columns, axis=1)
    
    # Create metadata with dataset name and index
    metadata_temp = pd.DataFrame({
        'Dataset': [dir_name] * len(X_temp),
        'Original_Index': range(len(X_temp))
    })
    
    combined_X = pd.concat([combined_X, X_temp], axis=0, ignore_index=True)
    combined_metadata = pd.concat([combined_metadata, metadata_temp], axis=0, ignore_index=True)
    
    # Store individual datasets for subject-specific cross-validation
    data_dict[dir_name] = {
        'features': X_temp,
        'labels': map_labels(df['Labels'], new_classes['2 levels']['order'])
    }
    
    # Extract the labels
    # df = pd.read_csv(f'data/logging/{dir_name}/whole_dataset_labels.csv')
    for merge in new_classes:
        Y_temp = map_labels(df['Labels'], new_classes[merge]['order'])
        new_classes[merge]['labels'].append(Y_temp)

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(combined_X)
combined_X = pd.DataFrame(X_normalized, columns=combined_X.columns)
combined_metadata = combined_metadata.reset_index(drop=True)  # Ensure metadata is aligned

# Normalize each dataset in data_dict using the same scaler
for dir_name, data in data_dict.items():
    data_dict[dir_name]['features'] = pd.DataFrame(
        scaler.transform(data['features']), 
        columns=data['features'].columns
    )

# Define the groups of features by channels
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
labels = pd.concat(new_classes['2 levels']['labels'], axis=0, ignore_index=True)

# Prepare base model probabilities using k-fold cross-validation
k = 10
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy', class_weight='balanced'),
    'svm': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    'logistic_regression': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
    'xgboost': XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=100, learning_rate=0.1, eval_metric='mlogloss', use_label_encoder=False, seed=42),
}

best_models = {}
base_model_probabilities = np.zeros((len(labels), 2))  # Binary classification
best_model_per_subject = defaultdict(lambda: defaultdict(list))


for channel, features in grouped_features.items():
    print(f"Evaluating models for channel: {channel}")
    
    best_model = None
    best_score = 0
    best_channel_probs = None
    
    for model_name, model in models.items():
        print(f"Testing model: {model_name}")
        
        channel_probs = np.zeros((len(labels), 2))
        cv_scores = []
        
        for train_index, test_index in kf.split(features, labels):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            model.fit(X_train, y_train)
            y_prob_test = model.predict_proba(X_test)
            
            channel_probs[test_index] = y_prob_test
            
            accuracy = accuracy_score(y_test, np.argmax(y_prob_test, axis=1))
            cv_scores.append(accuracy)
        
        # Calculate average score across folds
        average_score = np.mean(cv_scores)
        print(f"Average accuracy for {model_name} on channel {channel}: {average_score:.4f}")
        
        # Best model
        if average_score > best_score:
            best_score = average_score
            best_model = model_name
            best_channel_probs = channel_probs
    
    # Store the best model and its probabilities for this channel
    best_models[channel] = best_model
    base_model_probabilities += best_channel_probs
    print(f"Best model for channel {channel}: {best_model} with accuracy {best_score:.4f}")


# Visualize model selection
# for channel, features in grouped_features.items():
#     print(f"Evaluating models for channel: {channel}")
    
#     best_model = None
#     best_score = 0
#     best_channel_probs = None
    
#     # Dictionary to store the count of best models for each subject
#     model_counts_per_subject = defaultdict(int)

#     for model_name, model in models.items():
#         print(f"Testing model: {model_name}")
        
#         channel_probs = np.zeros((len(labels), 2))
#         cv_scores = []
        
#         for test_subject, test_data in data_dict.items():
#             test_index = combined_metadata[combined_metadata['Dataset'] == test_subject].index
#             X_test = test_data['features'][features.columns] 
#             y_test = test_data['labels']

#             # Combine data from all other subjects
#             train_data = combined_metadata[combined_metadata['Dataset'] != test_subject]
#             X_train = features.iloc[train_data.index]
#             y_train = labels[train_data.index]
            
#             model.fit(X_train, y_train)
#             y_prob_test = model.predict_proba(X_test)
            
#             channel_probs[test_index] = y_prob_test
            
#             accuracy = accuracy_score(y_test, np.argmax(y_prob_test, axis=1))
#             cv_scores.append(accuracy)
            
#             # Store the best model for each subject
#             if accuracy > model_counts_per_subject[test_subject]:
#                 best_model_per_subject[channel][test_subject] = model_name
#                 model_counts_per_subject[test_subject] = accuracy
        
#         # Calculate average score across folds
#         average_score = np.mean(cv_scores)
#         print(f"Average accuracy for {model_name} on channel {channel}: {average_score:.4f}")
        
#         # Best model
#         if average_score > best_score:
#             best_score = average_score
#             best_model = model_name
#             best_channel_probs = channel_probs
    
#     # Store the best model and its probabilities for this channel
#     best_models[channel] = best_model
#     base_model_probabilities += best_channel_probs
#     print(f"Best model for channel {channel}: {best_model} with accuracy {best_score:.4f}")

# for channel, subject_models in best_model_per_subject.items():
#     model_counts = pd.Series(subject_models.values()).value_counts()
    
#     plt.figure(figsize=(10, 6))
#     model_counts.plot(kind='bar', color='skyblue')
#     plt.title(f"Best Model Counts for Channel {channel}")
#     plt.xlabel('Model')
#     plt.ylabel('Number of Subjects')
#     plt.xticks(rotation=45)

# plt.show()
# exit(1)

final_probabilities = base_model_probabilities / len(grouped_features)
final_predictions = np.argmax(final_probabilities, axis=1)

# Evaluate final predictions
print(f"Soft Voting Accuracy: {accuracy_score(labels, final_predictions)}")
print(classification_report(labels, final_predictions))

# Calculate and plot normalized confusion matrix for soft voting
cm_final = confusion_matrix(labels, final_predictions)
cm_final_normalized = cm_final.astype('float') / cm_final.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 7))
sns.heatmap(cm_final_normalized, annot=True, fmt='.2%', cmap='Blues')
plt.title('Normalized Confusion Matrix for Soft Voting')
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()

onset_idx = {
    '1712094934_Nhan':16231,
    '1713985211_Mallory':3844,
    '1713999445_Wil': 1079,
    '1713293584_Jacob':30*250,
    '1713550894_AlexKagoda':25608,
    '1713820773_Andrew':8654,
    '1713994149_Saad':16231,
    '1714511822_Nathan':12742,
    '1713458084_Andrei':20*250
}

fs = 250

for dir_name in dir_names:

    plt.figure(figsize=(12, 8))  # Adjust figure size to fit a single plot
    if dir_name != '1713901362_Luke':
        
        dataset_indices = combined_metadata[combined_metadata['Dataset'] == dir_name].index
        
        # Initialize an array to store the sum of probabilities across channels
        avg_probs = np.zeros(len(dataset_indices))
        
        # Sum probabilities across channels
        for channel in channels.keys():
            channel_probs = final_probabilities[dataset_indices, 1]  # Class 1 probabilities for each channel
            avg_probs += channel_probs
        
        # Calculate the average probability across channels
        avg_probs /= len(channels)
        
        # Ensure probabilities are within [0, 1] before converting to percentage
        avg_probs = np.clip(avg_probs, 0, 1)
        avg_probs *= 100
        
        # Calculate the slope (first derivative) of the average probability
        slope = np.diff(avg_probs)
        
        # Plot the average probability
        plt.plot(avg_probs, label=f'Average Probability Across Channels')
        plt.axvline(x=int(onset_idx[dir_name]/fs), color='k', linestyle=':', label='Onset', linewidth=2)
        plt.text(int(onset_idx[dir_name]/fs), 1.05, 'Onset', color='k', ha='center', va='bottom', fontsize=10)
                
        plt.title(f"Average Probability of Belonging to Class 1 for Dataset: {dir_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Average Probability (%)")
        plt.legend()
        
        plt.tight_layout()

plt.show()


###################################################################################
###################################################################################
# Run the prediction on all data
import joblib
import os

models_folder = 'ML_Models'

if not os.path.exists(models_folder):
    os.makedirs(models_folder)


final_models = {}

for channel, features in grouped_features.items():
    print(f"Training final {best_models[channel]} model for channel: {channel}")
    
    model = models[best_models[channel]]
    model.fit(features, labels)
    model_filename = os.path.join(models_folder, f'{channel}_final_model_base_post_v6.joblib')
    final_models[channel] = model
    joblib.dump(model, model_filename)