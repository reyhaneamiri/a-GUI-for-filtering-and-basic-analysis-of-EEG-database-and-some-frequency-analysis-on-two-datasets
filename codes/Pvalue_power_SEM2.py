import numpy as np
import scipy.fftpack
from scipy.stats import ttest_ind, sem
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read EEG data from a text file
def read_eeg_file(filepath):
    try:
        return np.loadtxt(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

# Function to extract power of different frequency bands using FFT
def extract_band_power(data, Fs):
    n_channels, n_samples = data.shape
    freq_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    power = {band: np.zeros(n_channels) for band in freq_bands}
    
    for ch in range(n_channels):
        channel_data = data[ch]
        L = len(channel_data)
        NFFT = 2**np.ceil(np.log2(L)).astype(int)
        freqs = Fs / 2 * np.linspace(0, 1, NFFT // 2 + 1)
        xx = np.fft.fft(channel_data, n=NFFT)
        X = np.fft.fft(channel_data, n=NFFT)
        X = X[:NFFT // 2 + 1]
        X /= L

        for band, (low_freq, high_freq) in freq_bands.items():
            freq_indices = (freqs >= low_freq) & (freqs < high_freq)
            Xf = np.zeros_like(X)
            Xf[freq_indices] = X[freq_indices] * L
            
            filtered_signal = np.fft.ifft(Xf, n=NFFT).real * 2
            power_value = np.sum(filtered_signal**2) / len(xx)
            power_dB = 10 * np.log10(power_value)
            power[band][ch] = power_dB
    
    return power

# Parameters
Fs = 500  # Sampling frequency
n_channels = 15
original_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
desired_channel_order = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8']

# Directory where your files are located
data_directory = 'C:/Users/SURFACE/Desktop/proposal/age_filtered_signals/data'

# File paths
young_files = [os.path.join(data_directory, f'young_{i+1}') for i in range(14)]
old_files = [os.path.join(data_directory, f'old_{i+1}') for i in range(14)]

# Read EEG data
young_data = [read_eeg_file(f) for f in young_files if read_eeg_file(f) is not None]
old_data = [read_eeg_file(f) for f in old_files if read_eeg_file(f) is not None]

assert len(young_data) == 14, "Some young files are missing"
assert len(old_data) == 14, "Some old files are missing"

# Extract power for each band
young_powers = [extract_band_power(data, Fs) for data in young_data]
old_powers = [extract_band_power(data, Fs) for data in old_data]

# Reorder data according to the desired channel order
def reorder_data(data, original_order, new_order):
    index_map = {ch: i for i, ch in enumerate(original_order)}
    return [data[index_map[ch]] for ch in new_order]

# Function to calculate SEM for each group separately
def calculate_group_sem(group_powers):
    # sem_values = {}
    final_sem = {}
    for band in group_powers[0]:  # Iterate over frequency bands (keys in the dictionaries)
        # Collect power values for the specific band across all participants
        band_powers = np.array([participant[band] for participant in group_powers])
        # Calculate SEM across participants (axis=0) for each channel in the band
        #sem_values[band] = sem(band_powers, axis=0)
        mean_band_powers = np.mean(band_powers, axis=1)
        final_sem[band] = sem(mean_band_powers)

    return final_sem

# Calculate SEM for both groups
young_sem = calculate_group_sem(young_powers)
old_sem = calculate_group_sem(old_powers)

# Save p-values to an Excel file
p_values = {band: np.zeros(n_channels) for band in young_powers[0]}

for band in p_values.keys():
    for ch in range(n_channels):
        young_band_ch = [p[band][ch] for p in young_powers]
        old_band_ch = [p[band][ch] for p in old_powers]
        _, p_values[band][ch] = ttest_ind(young_band_ch, old_band_ch)

df_p_values = pd.DataFrame(p_values)
df_p_values.index = original_channels
df_p_values = df_p_values.loc[desired_channel_order]
df_p_values.to_excel('p_values1.xlsx')

print("P-values saved to p_values.xlsx")
# Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(df_p_values, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5)
plt.title('P-values Heatmap of Frequency Band Power Differences')
plt.xlabel('Frequency Band')
plt.ylabel('Channel')
plt.show()


def plot_power_comparison(young_powers, old_powers, freq_band, original_channels, desired_order, young_sem, old_sem):
    # Calculate mean power for each group and each channel
    young_band_power = np.mean([p[freq_band] for p in young_powers], axis=0)
    old_band_power = np.mean([p[freq_band] for p in old_powers], axis=0)
    
    # Reorder the data
    young_band_power = reorder_data(young_band_power, original_channels, desired_order)
    old_band_power = reorder_data(old_band_power, original_channels, desired_order)
    
    # # Reorder SEMs
    # young_band_sem = reorder_data(young_sem[freq_band], original_channels, desired_order)
    # old_band_sem = reorder_data(old_sem[freq_band], original_channels, desired_order)
    
    x = np.arange(len(desired_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, young_band_power, width, label='Young', alpha=0.7)
    bars2 = ax.bar(x + width/2, old_band_power, width, label='Old', alpha=0.7)

    ax.set_xlabel('Channels')
    ax.set_ylabel('Power')
    ax.set_title(f'Comparison of {freq_band.capitalize()} Band Power Between Young and Old Groups')
    ax.set_xticks(x)
    ax.set_xticklabels(desired_order, rotation=45)
    ax.legend()

    # Conclusion text based on the comparison of means and SEM
    conclusion_text = ''
    if np.mean(young_band_power) > np.mean(old_band_power):
        conclusion_text = f'Young group has higher {freq_band} power on average'
    else:
        conclusion_text = f'Old group has higher {freq_band} power on average'
    # # Conclusion text based on the comparison of SEM
    # sem_diff = young_sem[freq_band] - old_sem[freq_band]
    # if sem_diff < 0:
    #     conclusion_text += f'\nYoung group data is more consistent (lower SEM) for {freq_band} band'
    # elif sem_diff > 0:
    #     conclusion_text += f'\nOld group data is more consistent (lower SEM) for {freq_band} band'
    # else:
    #     conclusion_text += f'\nSEM values are similar for {freq_band} band across both groups'

    # conclusion_text += f'\n(Young SEM: {young_sem[freq_band]:.2f}, Old SEM: {old_sem[freq_band]:.2f})'
    
    
    # # Add annotations to the plot for mean values
    # for i in range(len(desired_order)):
    #     ax.text(x[i] - width/2, young_band_power[i] + 0.5, f'{young_band_power[i]:.2f}', ha='center', va='bottom', fontsize=10)
    #     ax.text(x[i] + width/2, old_band_power[i] + 0.5, f'{old_band_power[i]:.2f}', ha='center', va='bottom', fontsize=10)

    ax.text(0.5, 0.9, conclusion_text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
for band in p_values.keys():
    plot_power_comparison(young_powers, old_powers, band, original_channels, desired_channel_order, young_sem, old_sem)
