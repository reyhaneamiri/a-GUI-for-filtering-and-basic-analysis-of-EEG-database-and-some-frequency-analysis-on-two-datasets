import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, Scrollbar, IntVar, StringVar, OptionMenu
import webbrowser
from tkinter.filedialog import asksaveasfilename, askopenfilename
import numpy as np
import mne
import matplotlib.pyplot as plt



# Load the Excel file
df = pd.read_excel('C:/Users/SURFACE/Desktop/proposal/database_edit5.xlsx')

# Replace "not mentioned" strings with NaN
df.replace("not mentioned", pd.NA, inplace=True)

# Convert columns containing numeric data to numeric data types
numeric_columns = ['min_age', 'max_age', 'Fs', 'number of participants']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Abbreviations for the channel regions
abbreviations = {
    "frontal": "F",
    "central": "C",
    "parietal": "P",
    "occipital": "O",
    "temporal": "T"
}

# Global variables
original_raw = None
preprocessed_raw = None

# Function to load EEG data
def load_eeg():
    global original_raw, preprocessed_raw
    file_path = askopenfilename(filetypes=[("EEG Files", "*.fif;*.edf;*.bdf;*.vhdr;*.eeg;*.set;*.gdf;*.cnt;*.mff;*.nxe;*.ns;*.mef")])
    if file_path:
        original_raw = mne.io.read_raw(file_path, preload=True)
        preprocessed_raw = None
        metadata_frame = tk.Toplevel(root)
        metadata_frame.title("Metadata")

        # Extract metadata
        info = original_raw.info
        n_channels = len(info['ch_names'])
        ch_names = ', '.join(info['ch_names'])
        length_seconds = original_raw.n_times / info['sfreq']
        sfreq = info['sfreq']

        # Display metadata
        tk.Label(metadata_frame, text=f"Number of Channels: {n_channels}").pack()
        tk.Label(metadata_frame, text=f"Channel Names: {ch_names}").pack()
        tk.Label(metadata_frame, text=f"Length of Signal: {length_seconds:.2f} seconds").pack()
        tk.Label(metadata_frame, text=f"Sampling Frequency: {sfreq} Hz").pack()

        # Buttons for preprocessing and processing
        tk.Button(metadata_frame, text="Preprocessing", command=lambda: open_preprocessing_page(original_raw)).pack(side=tk.LEFT, fill=tk.X, padx=5)
        tk.Button(metadata_frame, text="Processing", command=lambda: open_processing_page(original_raw)).pack(side=tk.LEFT, fill=tk.X, padx=5)

def open_preprocessing_page(raw):
    preprocessing_frame = tk.Toplevel(root)
    preprocessing_frame.title("Preprocessing")

    def apply_preprocessing():
        global preprocessed_raw
        if preprocessed_raw is None:
            preprocessed_raw = original_raw.copy()
        # Apply selected preprocessing steps
        if high_pass_var.get():
            preprocessed_raw = preprocessed_raw.copy().filter(l_freq=float(high_pass_entry.get()), h_freq=None)
        if low_pass_var.get():
            preprocessed_raw = preprocessed_raw.copy().filter(l_freq=None, h_freq=float(low_pass_entry.get()))
        if notch_var.get():
            preprocessed_raw = preprocessed_raw.copy().notch_filter(freqs=[float(notch_entry.get())])
        if dc_offset_var.get():
            preprocessed_raw = preprocessed_raw.copy().filter(l_freq=0.1, h_freq=None)
        if resample_var.get():
            preprocessed_raw = preprocessed_raw.copy().resample(sfreq=float(resample_entry.get()))
        # Save preprocessed data
        save_path = asksaveasfilename(defaultextension=".fif", filetypes=[("FIF Files", "*.fif")])
        if save_path:
            preprocessed_raw.save(save_path, overwrite=True)
            messagebox.showinfo("Success", f"Preprocessed data saved to {save_path}")

    # High-pass filter
    high_pass_var = IntVar()
    tk.Checkbutton(preprocessing_frame, text="High-pass filter", variable=high_pass_var).pack()
    tk.Label(preprocessing_frame, text="High-pass cutoff (Hz):").pack()
    high_pass_entry = tk.Entry(preprocessing_frame)
    high_pass_entry.pack()

    # Low-pass filter
    low_pass_var = IntVar()
    tk.Checkbutton(preprocessing_frame, text="Low-pass filter", variable=low_pass_var).pack()
    tk.Label(preprocessing_frame, text="Low-pass cutoff (Hz):").pack()
    low_pass_entry = tk.Entry(preprocessing_frame)
    low_pass_entry.pack()

    # Notch filter
    notch_var = IntVar()
    tk.Checkbutton(preprocessing_frame, text="Notch filter", variable=notch_var).pack()
    tk.Label(preprocessing_frame, text="Notch frequency (Hz):").pack()
    notch_entry = tk.Entry(preprocessing_frame)
    notch_entry.pack()

    # DC offset
    dc_offset_var = IntVar()
    tk.Checkbutton(preprocessing_frame, text="Remove DC offset", variable=dc_offset_var).pack()
    # Resample
    resample_var = IntVar()
    tk.Checkbutton(preprocessing_frame, text="Change sampling frequency", variable=resample_var).pack()
    tk.Label(preprocessing_frame, text="New sampling frequency (Hz):").pack()
    resample_entry = tk.Entry(preprocessing_frame)
    resample_entry.pack()
    # Apply button
    tk.Button(preprocessing_frame, text="Apply", command=apply_preprocessing).pack()

def open_processing_page(raw):
    global original_raw, preprocessed_raw
    processing_frame = tk.Toplevel(root)
    processing_frame.title("Processing")

    def select_all_bands(event=None):
        selected_indices = bands_listbox.curselection()
        if 'All' in [bands_listbox.get(i) for i in selected_indices]:
            if bands_listbox.get(0, tk.END)[-1] == 'All':
                bands_listbox.selection_set(0, tk.END)
            else:
                bands_listbox.selection_clear(0, tk.END)

    def select_all_channels(event=None):
        selected_indices = channels_listbox.curselection()
        if 'All' in [channels_listbox.get(i) for i in selected_indices]:
            if channels_listbox.get(0, tk.END)[-1] == 'All':
                channels_listbox.selection_set(0, tk.END)
            else:
                channels_listbox.selection_clear(0, tk.END)

    def plot_channels():
        if preprocessed_raw is not None:
            raw_to_process = preprocessed_raw
        elif original_raw is not None:
            raw_to_process = original_raw
        else:
            messagebox.showerror("Error", "No EEG data loaded.")
            return

        selected_channels = [channels_listbox.get(idx) for idx in channels_listbox.curselection() if channels_listbox.get(idx) != 'All']
        if not selected_channels:
            messagebox.showwarning("Warning", "No channels selected.")
            return
        raw_to_process.plot(n_channels=len(selected_channels), picks=selected_channels, show=True)

    def plot_frequency_bands():
        if preprocessed_raw is not None:
            raw_to_process = preprocessed_raw
        elif original_raw is not None:
            raw_to_process = original_raw
        else:
            messagebox.showerror("Error", "No EEG data loaded.")
            return

        selected_band_indices = bands_listbox.curselection()
        selected_bands = [bands[idx] for idx in selected_band_indices if bands[idx] != 'All']
        if 'All' in selected_bands:
            selected_bands = bands[:-1]  # Select all except 'All'

        if not selected_bands:
            messagebox.showwarning("Warning", "No frequency bands selected.")
            return

        selected_channels = [channels_listbox.get(idx) for idx in channels_listbox.curselection() if channels_listbox.get(idx) != 'All']
        if 'All' in selected_channels:
            selected_channels = channels_listbox.get(0, tk.END)[:-1]  # Select all except 'All'

        if not selected_channels:
            messagebox.showwarning("Warning", "No channels selected.")
            return

        time = {band: [] for band in selected_bands}
        for band in selected_bands:
            low_freq, high_freq = bands_dict[band]
            for ch in selected_channels:
                raw_copy = raw_to_process.copy().pick(picks=[ch])
                data = raw_copy.get_data(picks=[ch])
                L = data.shape[1]
                NFFT = 2**np.ceil(np.log2(L)).astype(int)  # Length of FFT
                Fs = raw_to_process.info['sfreq']
                freqs = raw_to_process.info['sfreq'] / 2 * np.linspace(0, 1, NFFT // 2 + 1)
                time = np.linspace(0, L / Fs, L)
                len_time = len(time)

                # Compute FFT
                xx = np.fft.fft(data, n=NFFT, axis=1) 
                X = np.fft.fft(data, n=NFFT, axis=1) 
                X = X[:, :NFFT // 2 + 1]  # Consider only the first half (up to Nyquist frequency)
                X = X / L
                len_xx = xx.shape[1]
                f_len = min(len_xx, len_time)
                # Frequency band filtering
                
                Xf = np.zeros_like(X)
                freq_indices = (freqs >= low_freq) & (freqs < high_freq)
                

                Xf[:, freq_indices] = X[:, freq_indices] * L  # Apply the filter and scaling
                
                
                filtered_signal = np.fft.ifft(Xf, n=NFFT, axis=1).real * 2
                filtered_signal =  filtered_signal[: , :f_len]
                filtered_signal = filtered_signal.flatten()
                fig, ax = plt.subplots()
                ax.plot(time, filtered_signal)
                ax.set_title(f'{band} Band - Channel: {ch}')
                ax.set_ylabel('Amplitude')
                plt.show()

        

    def power_of_frequency_bands():
        if preprocessed_raw is not None:
            raw_to_process = preprocessed_raw
        elif original_raw is not None:
            raw_to_process = original_raw
        else:
            messagebox.showerror("Error", "No EEG data loaded.")
            return

        selected_band_indices = bands_listbox.curselection()
        selected_bands = [bands[idx] for idx in selected_band_indices if bands[idx] != 'All']
        if 'All' in selected_bands:
            selected_bands = bands[:-1]  # Select all except 'All'

        if not selected_bands:
            messagebox.showwarning("Warning", "No frequency bands selected.")
            return

        selected_channels = [channels_listbox.get(idx) for idx in channels_listbox.curselection() if channels_listbox.get(idx) != 'All']
        if 'All' in selected_channels:
            selected_channels = channels_listbox.get(0, tk.END)[:-1]  # Select all except 'All'

        if not selected_channels:
            messagebox.showwarning("Warning", "No channels selected.")
            return

        band_powers = {band: [] for band in selected_bands}
        for band in selected_bands:
            low_freq, high_freq = bands_dict[band]

            for ch in selected_channels:
                raw_copy = raw_to_process.copy().pick(picks=[ch])
                data = raw_copy.get_data(picks=[ch])
                L = data.shape[1]
                NFFT = 2**np.ceil(np.log2(L)).astype(int)  # Length of FFT
                freqs = raw_to_process.info['sfreq'] / 2 * np.linspace(0, 1, NFFT // 2 + 1)
                
                # Compute FFT
                xx = np.fft.fft(data, n=NFFT, axis=1) 
                X = np.fft.fft(data, n=NFFT, axis=1) 
                X = X[:, :NFFT // 2 + 1]  # Consider only the first half (up to Nyquist frequency)
                X = X / L
            
                # Frequency band filtering
                
                Xf = np.zeros_like(X)
                freq_indices = (freqs >= low_freq) & (freqs < high_freq)
                

                Xf[:, freq_indices] = X[:, freq_indices] * L  # Apply the filter and scaling
                
                
                filtered_signal = np.fft.ifft(Xf, n=NFFT, axis=1).real * 2
            
                # Normalize power calculation by length of FFT result
                
                length_xx = xx.shape[1]
                power = np.sum(filtered_signal**2, axis=1) / length_xx
                power_dB = 10 * np.log10(power)  
            
                band_powers[band].append((power_dB))


        save_path = asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if save_path:
            df_band_powers = pd.DataFrame(band_powers, index=selected_channels)
            df_band_powers.to_excel(save_path)
            messagebox.showinfo("Success", f"Band power data saved to {save_path}")

    def plot_fft_of_channels():
        if preprocessed_raw is not None:
            raw_to_process = preprocessed_raw
        elif original_raw is not None:
            raw_to_process = original_raw
        else:
            messagebox.showerror("Error", "No EEG data loaded.")
            return

        selected_channels = [channels_listbox.get(idx) for idx in channels_listbox.curselection() if channels_listbox.get(idx) != 'All']
        if 'All' in selected_channels:
            selected_channels = channels_listbox.get(0, tk.END)[:-1]  # Select all except 'All'

        if not selected_channels:
            messagebox.showwarning("Warning", "No channels selected.")
            return

        try:
            low_freq = float(low_freq_entry.get())
            high_freq = float(high_freq_entry.get())
        except ValueError:
            messagebox.showwarning("Warning", "Please enter valid frequency range.")
            return

        nyquist_freq = raw_to_process.info['sfreq'] / 2
        if low_freq > nyquist_freq or high_freq > nyquist_freq:
            messagebox.showwarning("Warning", f"Frequencies must be less than or equal to Nyquist frequency ({nyquist_freq} Hz).")
            return

        for ch in selected_channels:
            data = raw_to_process.get_data(picks=[ch])[0]
            sfreq = raw_to_process.info['sfreq']
            n = len(data)
            freqs = np.fft.rfftfreq(n, d=1/sfreq)
            fft_data = np.fft.rfft(data)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(freqs, np.abs(fft_data))
            ax.set_xlim([low_freq, high_freq])
            ax.set_title(f'FFT of Channel: {ch}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            plt.show()

    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'All']
    bands_dict = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 80)}

    tk.Label(processing_frame, text="Select Frequency Bands:").pack()
    bands_listbox = tk.Listbox(processing_frame, selectmode=tk.MULTIPLE, exportselection=False)
    bands_listbox.pack()
    for band in bands:
        bands_listbox.insert(tk.END, band)
    bands_listbox.bind('<<ListboxSelect>>', select_all_bands)

    tk.Label(processing_frame, text="Select Channels:").pack()
    channels_listbox = tk.Listbox(processing_frame, selectmode=tk.MULTIPLE, exportselection=False)
    channels_listbox.pack()
    if original_raw is not None:
        for ch in original_raw.info['ch_names']:
            channels_listbox.insert(tk.END, ch)
    channels_listbox.insert(tk.END, 'All')
    channels_listbox.bind('<<ListboxSelect>>', select_all_channels)

    tk.Label(processing_frame, text="Low Frequency (Hz):").pack()
    low_freq_entry = tk.Entry(processing_frame)
    low_freq_entry.pack()
    tk.Label(processing_frame, text="High Frequency (Hz):").pack()
    high_freq_entry = tk.Entry(processing_frame)
    high_freq_entry.pack()

    tk.Button(processing_frame, text="Plot Channels", command=plot_channels).pack()
    tk.Button(processing_frame, text="Plot Frequency Bands", command=plot_frequency_bands).pack()
    tk.Button(processing_frame, text="Power of Frequency Bands", command=power_of_frequency_bands).pack()
    tk.Button(processing_frame, text="Plot FFT of Channels", command=plot_fft_of_channels).pack()

def filter_data():
    # Get the values of the checkbuttons to determine which feature to filter on
    age_enabled = age_check_var.get()
    fs_enabled = fs_check_var.get()
    channels_enabled = channels_check_var.get()
    participants_enabled = participants_check_var.get()
    gender_enabled = gender_check_var.get()  
    channel_names_enabled = channel_names_check_var.get()
    
    # Check if none of the filters are enabled
    if not (age_enabled or fs_enabled or channels_enabled or participants_enabled or gender_enabled or channel_names_enabled):
        display_all_data()
        return
    
    # Get the range for the enabled feature
    if age_enabled:
        min_age = int(age_min_entry.get())
        max_age = int(age_max_entry.get())
    else:
        min_age = 0
        max_age = float('inf')
    
    if fs_enabled:
        min_fs = int(fs_min_entry.get())
        max_fs = int(fs_max_entry.get())
    else:
        min_fs = 0
        max_fs = float('inf')
    
    if channels_enabled:
        min_channels = int(channels_min_entry.get())
        max_channels = int(channels_max_entry.get())
    else:
        min_channels = 0
        max_channels = float('inf')
    
    if participants_enabled:
        min_participants = int(participants_min_entry.get())
        max_participants = int(participants_max_entry.get())
    else:
        min_participants = 0
        max_participants = float('inf')
    
    if gender_enabled:
        gender_value = gender_var.get()  # Get value from dropdown menu
    else:
        gender_value = None
    
    # Get selected channel abbreviations
    selected_channels = [abbreviations[channel] for channel, var in zip(["frontal", "central", "parietal", "occipital", "temporal"], channel_vars) if var.get()]
    hd_eeg_enabled = hdeeg_check_var.get()  # Check if HD-EEG is selected
    
    def check_channels(channels):
        channel_list = eval(channels)  # Convert the string representation of the list back to a list
        channel_checks = all(any(abbrev in channel for channel in channel_list) for abbrev in selected_channels)
        hd_eeg_checks = any(channel.startswith("E") for channel in channel_list) if hd_eeg_enabled else True
        return channel_checks and hd_eeg_checks

    # Filter data based on the enabled feature and selected channels
    filtered_data = df[
        ((df['min_age'] >= min_age) & (df['max_age'] <= max_age) if age_enabled else True) &
        ((df['Fs'] >= min_fs) & (df['Fs'] <= max_fs) if fs_enabled else True) &
        ((df['number of channels'].apply(lambda x: eval(x) if pd.notna(x) else [])  
          .apply(lambda channels: any(min_channels <= val <= max_channels for val in channels))) if channels_enabled else True) &
        ((df['number of participants'] >= min_participants) & (df['number of participants'] <= max_participants) if participants_enabled else True) &
        ((df['Gender'].apply(lambda x: eval(x) if pd.notna(x) else [])  
          .apply(lambda genders: gender_value in genders) if pd.notna(gender_value) else False) if gender_enabled else True) &
        (df['names of channels'].apply(check_channels) if (selected_channels or hd_eeg_enabled) else True)  # Filter by selected channels and HD-EEG
    ]

    if not filtered_data.empty:  # Check if filtered data is not empty
        # Get the data links and study names corresponding to the filtered data
        data_links = filtered_data['Ref Link'].tolist()
        study_names = filtered_data['Study name'].tolist()
        row_numbers = filtered_data['No.'].tolist()
        display_links(data_links, study_names, row_numbers, filtered_data)
    else:
        tk.messagebox.showinfo("No Data", "No data available for the selected filters.")

def display_all_data():
    # Get all data links and study names
    data_links = df['Ref Link'].tolist()
    study_names = df['Study name'].tolist()
    row_numbers = df['No.'].tolist()
    if data_links:
        display_links(data_links, study_names, row_numbers, df)
    else:
        tk.messagebox.showinfo("No Data", "No data available.")

def display_links(links, names, numbers, filtered_data):
    # Create a new window to display the links and study names
    link_window = tk.Toplevel(root)
    link_window.title("Filtered Data Links")
    
    # Create a Listbox to display the data
    listbox = Listbox(link_window, width=100)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Create a Scrollbar and attach it to the Listbox
    scrollbar = Scrollbar(link_window, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    # Add data to the Listbox
    for number, link, name in zip(numbers, links, names):
        listbox.insert(tk.END, f"{number}. {name}: {link}")
    
    # Function to open the selected link in the default web browser
    def open_selected_link(event):
        selected_index = listbox.curselection()
        if selected_index:
            selected_link = links[selected_index[0]]
            webbrowser.open_new(selected_link)
    
    # Bind the Double-Click event to open the selected link
    listbox.bind("<Double-Button-1>", open_selected_link)

    # Function to save the filtered data to an Excel file
    def save_to_excel():
        # Ask user for the file save location
        file_path = asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            # Save the filtered data to the specified file
            filtered_data.to_excel(file_path, index=False)
            tk.messagebox.showinfo("Success", f"Data saved to {file_path}")

    # Create a "Download Data" button
    download_button = tk.Button(link_window, text="Download Data", command=save_to_excel)
    download_button.pack(side=tk.BOTTOM)

# Create the GUI
root = tk.Tk()
root.title("Database Filter")

# Checkbuttons for enabling/disabling features
age_check_var = IntVar()
age_check = tk.Checkbutton(root, text="Age", variable=age_check_var)
age_check.grid(row=0, column=0)

fs_check_var = IntVar()
fs_check = tk.Checkbutton(root, text="Sampling Frequency", variable=fs_check_var)
fs_check.grid(row=1, column=0)

channels_check_var = IntVar()
channels_check = tk.Checkbutton(root, text="Number of Channels", variable=channels_check_var)
channels_check.grid(row=2, column=0)

participants_check_var = IntVar()
participants_check = tk.Checkbutton(root, text="Number of Participants", variable=participants_check_var)
participants_check.grid(row=3, column=0)

gender_check_var = IntVar()  
gender_check = tk.Checkbutton(root, text="Gender", variable=gender_check_var)
gender_check.grid(row=4, column=0)

channel_names_check_var = IntVar()
channel_names_check = tk.Checkbutton(root, text="Names of Channels", variable=channel_names_check_var)
channel_names_check.grid(row=5, column=0)

# Entry fields for user input
age_min_label = tk.Label(root, text="Min Age:")
age_min_label.grid(row=0, column=1)
age_min_entry = tk.Entry(root)
age_min_entry.grid(row=0, column=2)

age_max_label = tk.Label(root, text="Max Age:")
age_max_label.grid(row=0, column=3)
age_max_entry = tk.Entry(root)
age_max_entry.grid(row=0, column=4)

fs_min_label = tk.Label(root, text="Min Sampling Frequency (Hz):")
fs_min_label.grid(row=1, column=1)
fs_min_entry = tk.Entry(root)
fs_min_entry.grid(row=1, column=2)

fs_max_label = tk.Label(root, text="Max Sampling Frequency (Hz):")
fs_max_label.grid(row=1, column=3)
fs_max_entry = tk.Entry(root)
fs_max_entry.grid(row=1, column=4)

channels_min_label = tk.Label(root, text="Min Number of Channels:")
channels_min_label.grid(row=2, column=1)
channels_min_entry = tk.Entry(root)
channels_min_entry.grid(row=2, column=2)

channels_max_label = tk.Label(root, text="Max Number of Channels:")
channels_max_label.grid(row=2, column=3)
channels_max_entry = tk.Entry(root)
channels_max_entry.grid(row=2, column=4)

participants_min_label = tk.Label(root, text="Min Number of Participants:")
participants_min_label.grid(row=3, column=1)
participants_min_entry = tk.Entry(root)
participants_min_entry.grid(row=3, column=2)

participants_max_label = tk.Label(root, text="Max Number of Participants:")
participants_max_label.grid(row=3, column=3)
participants_max_entry = tk.Entry(root)
participants_max_entry.grid(row=3, column=4)

# Dropdown menu for gender selection
gender_label = tk.Label(root, text="Gender:")
gender_label.grid(row=4, column=1)
gender_var = StringVar(root)
gender_dropdown = OptionMenu(root, gender_var, "Male", "Female")
gender_dropdown.grid(row=4, column=2)

# Checkbuttons for selecting channels
channel_labels = ["Frontal", "Central", "Parietal", "Occipital", "Temporal"]
channel_vars = [IntVar() for _ in channel_labels]
for i, label in enumerate(channel_labels):
    chk = tk.Checkbutton(root, text=label, variable=channel_vars[i])
    chk.grid(row=5, column=1+i)

# Checkbutton for HD-EEG
hdeeg_check_var = IntVar()
hdeeg_check = tk.Checkbutton(root, text="HD-EEG", variable=hdeeg_check_var)
hdeeg_check.grid(row=5, column=6)

# Filter button
filter_button = tk.Button(root, text="Filter Data", command=filter_data)
filter_button.grid(row=7, column=0, columnspan=7)

# Upload EEG button
formats_label = tk.Label(root, text="Accepted EEG formats: .fif, .edf, .bdf, .vhdr, .eeg, .set, .gdf, .cnt, .mff, .nxe, .ns, .mef")
formats_label.grid(row=8, column=0, columnspan=7, pady=(0, 10))  # Adjust the row and pady for spacing
upload_button = tk.Button(root, text="Upload EEG Signal", command=load_eeg)
upload_button.grid(row=9, column=0, columnspan=7, pady=20)

# Start the Tkinter main loop
root.mainloop()
