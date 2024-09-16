import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.stats import spearmanr

# Function to read EEG data from a text file using MNE
def read_eeg_file(filepath, Fs, ch_names):
    try:
        data = np.loadtxt(filepath)
        info = mne.create_info(ch_names=ch_names, sfreq=Fs, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        return raw
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

# Function to extract timeseries of different frequency bands using FFT
def extract_band(data, Fs):
    n_channels, n_samples = data.shape
    freq_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    band_time = {band: np.zeros((n_channels, n_samples)) for band in freq_bands}
    
    for ch in range(n_channels):
        channel_data = data[ch]
        L = len(channel_data)
        NFFT = 2**np.ceil(np.log2(L)).astype(int)
        freqs = Fs / 2 * np.linspace(0, 1, NFFT // 2 + 1)
        
        X = np.fft.fft(channel_data, n=NFFT)
        X = X[:NFFT // 2 + 1]
        X /= L

        for band, (low_freq, high_freq) in freq_bands.items():
            freq_indices = (freqs >= low_freq) & (freqs < high_freq)
            Xf = np.zeros_like(X)
            Xf[freq_indices] = X[freq_indices] * L
            
            filtered_signal = np.fft.ifft(Xf, n=NFFT).real * 2
            
            band_time[band][ch] = filtered_signal[:n_samples]  # Keep the same length as original data
    
    return band_time

# Function to calculate the functional connectivity matrix using extracted bands
def calculate_functional_connectivity(band_data, freq_bands):
    # Initialize a dictionary to store connectivity matrices for each frequency band
    connectivity = {band: np.zeros((band_data[band].shape[0], band_data[band].shape[0])) for band in freq_bands}
    
    for band in freq_bands.keys():
        # Iterate over each pair of channels to compute connectivity
        for i in range(band_data[band].shape[0]):  # i is the index of the first channel
            for j in range(band_data[band].shape[0]):  # j is the index of the second channel
                if i != j:
                    # Compute Spearman correlation between the two channels for the given band
                    connectivity[band][i, j] = spearmanr(band_data[band][i], band_data[band][j])[0]
                else:
                    # Set diagonal elements (self-connectivity) to 1
                    connectivity[band][i, j] = 1
    
    return connectivity


# Function to save connectivity matrices and generate heatmaps
def save_connectivity_matrices(connectivity, save_path, group, person_id):
    os.makedirs(save_path, exist_ok=True)
    for band, matrix in connectivity.items():
        spearman_file = os.path.join(save_path, f'{group}_person_{person_id}_{band}_spearman.csv')
        pd.DataFrame(matrix).to_csv(spearman_file, index=False, header=False)
        
        # Generate and save heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=False, cmap='coolwarm', cbar=True)
        plt.title(f'{group.capitalize()} Person {person_id} - {band.capitalize()} Band Connectivity')
        plt.savefig(os.path.join(save_path, f'{group}_person_{person_id}_{band}_heatmap.png'))
        plt.close()

# Parameters
Fs = 500  # Sampling frequency
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']

# Directory where your files are located
data_directory = 'C:/Users/SURFACE/Desktop/proposal/age_filtered_signals/data'

# File paths
young_files = [os.path.join(data_directory, f'young_{i+1}') for i in range(14)]
old_files = [os.path.join(data_directory, f'old_{i+1}') for i in range(14)]

# Frequency bands
freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

# Read EEG data
young_data = [read_eeg_file(f, Fs, channels) for f in young_files if read_eeg_file(f, Fs, channels) is not None]
old_data = [read_eeg_file(f, Fs, channels) for f in old_files if read_eeg_file(f, Fs, channels) is not None]

# Ensure that we have the expected number of files
assert len(young_data) == 14, "Some young files are missing"
assert len(old_data) == 14, "Some old files are missing"

# Calculate functional connectivity for each person in both groups
young_connectivity = []
old_connectivity = []

for raw in young_data:
    band_time = extract_band(raw.get_data(), Fs)
    conn = calculate_functional_connectivity(band_time, freq_bands)
    young_connectivity.append(conn)

for raw in old_data:
    band_time = extract_band(raw.get_data(), Fs)
    conn = calculate_functional_connectivity(band_time, freq_bands)
    old_connectivity.append(conn)

# Save connectivity matrices and heatmaps
save_path = 'C:/Users/SURFACE/Desktop/proposal/connectivity_matrices'
for i, conn in enumerate(young_connectivity):
    save_connectivity_matrices(conn, save_path, 'young', i + 1)
for i, conn in enumerate(old_connectivity):
    save_connectivity_matrices(conn, save_path, 'old', i + 1)

print("Functional connectivity matrices and heatmaps saved.")
