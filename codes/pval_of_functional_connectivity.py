import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Number of channels and their order
original_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
desired_channel_order = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8']
n_channels = len(desired_channel_order)

# Frequency bands
freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

# Function to reorder the matrix based on the desired channel order
def reorder_matrix(matrix, original_order, new_order):
    index_map = {ch: i for i, ch in enumerate(original_order)}
    reordered_matrix = np.zeros_like(matrix)
    
    for i, ch1 in enumerate(new_order):
        for j, ch2 in enumerate(new_order):
            reordered_matrix[i, j] = matrix[index_map[ch1], index_map[ch2]]
    
    return reordered_matrix

# Function to read the functional connectivity matrix from CSV
def read_functional_connectivity_matrix(filepath):
    df = pd.read_csv(filepath, header=None)
    matrix = df.to_numpy()
    return matrix

# Function to calculate p-values from the functional connectivity matrices
def calculate_p_values(young_matrices, old_matrices):
    bands = list(freq_bands.keys())
    p_values = {band: np.zeros((n_channels, n_channels)) for band in bands}

    for band in bands:
        young_values = np.array([m for m in young_matrices[band] if m is not None])
        old_values = np.array([m for m in old_matrices[band] if m is not None])

        if young_values.size > 0 and old_values.size > 0:
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    young_pair = young_values[:, i, j]
                    old_pair = old_values[:, i, j]

                    if len(young_pair) > 1 and len(old_pair) > 1:  # Ensure there are enough values for t-test
                        _, p_val = ttest_ind(young_pair, old_pair)
                    else:
                        p_val = np.nan  # Not enough data for t-test

                    p_values[band][i, j] = p_val
                    p_values[band][j, i] = p_val  # Matrix is symmetric

    return p_values

# Function to save and plot p-values heatmap
def save_and_plot_heatmaps(p_values, save_path):
    os.makedirs(save_path, exist_ok=True)
    bands = list(p_values.keys())
    
    for band in bands:
        p_val_matrix = reorder_matrix(p_values[band], original_channels, desired_channel_order)
        
        # Mask the lower triangle
        mask = np.triu(np.ones_like(p_val_matrix, dtype=bool), k=1)

        # Create annotation matrix with asterisks for p-values < 0.05
        annotations = np.full(p_val_matrix.shape, '', dtype=object)
        annotations[p_val_matrix < 0.05] = '*'
        
        # Save p-values matrix
        pd.DataFrame(p_val_matrix).to_csv(os.path.join(save_path, f'{band}_p_values.csv'), index=False, header=False)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(p_val_matrix, mask=mask, annot=annotations, fmt='', cmap='coolwarm', cbar=True, linewidths=0.5, vmin=0, vmax=0.5,
                    xticklabels=desired_channel_order, yticklabels=desired_channel_order)
        plt.title(f'{band.capitalize()} Band - P-Values Heatmap (Upper Triangle)')
        plt.xlabel('Channel')
        plt.ylabel('Channel')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{band}_p_values_heatmap_upper_triangle.png'))
        plt.close()

# Path where your functional connectivity matrices are saved
matrix_path = 'C:/Users/SURFACE/Desktop/proposal/connectivity_matrices'
save_path = 'C:/Users/SURFACE/Desktop/proposal/p_values_heatmaps'

# Load matrices
young_matrices = {band: [read_functional_connectivity_matrix(os.path.join(matrix_path, f'young_person_{i+1}_{band}_spearman.csv')) for i in range(14)] for band in freq_bands.keys()}
old_matrices = {band: [read_functional_connectivity_matrix(os.path.join(matrix_path, f'old_person_{i+1}_{band}_spearman.csv')) for i in range(14)] for band in freq_bands.keys()}

# Calculate p-values
p_values = calculate_p_values(young_matrices, old_matrices)

# Save and plot heatmaps
save_and_plot_heatmaps(p_values, save_path)

print("P-values heatmaps (upper triangle) have been saved.")
