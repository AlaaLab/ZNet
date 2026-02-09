#######################################################################################
# Author: Franny Dean
# Script: ecg_utils.py
# Function: helpers for reading ecg and training ecg based models
# Date: 02/06/2026
#######################################################################################
from torch.utils.data import Dataset
import torch
from pathlib import Path
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

class ECGReader:
    """Read ECG data from PhysioNet format (.hea and .mat files)."""
    
    def __init__(self, base_path):
        """
        Initialize ECG reader for a single record.
        
        Args:
            base_path (str or Path): Base path to ECG record (without suffix).
        """
        self.base_path = Path(base_path)
        self.header_path = self.base_path.with_suffix('.hea')
        self.data_path = self.base_path.with_suffix('.mat')
        
        self.signal = None
        self.sampling_rate = None
        self.n_samples = None
        self.n_channels = None
        self.channel_names = []
        self.units = []
        self.gain = []
        self.baseline = []
        
    def read_header(self):
        """
        Parse the .hea header file.
        
        Returns:
            dict: Metadata including sampling rate, channel names, and demographics.
        """
        with open(self.header_path, 'r') as f:
            lines = f.readlines()
        
        # First line: record info
        first_line = lines[0].split()
        self.n_channels = int(first_line[1])
        self.sampling_rate = float(first_line[2])
        self.n_samples = int(first_line[3])
        
        # Initialize metadata
        self.age = None
        self.sex = None
        self.dx_codes = []

        
        # Subsequent lines: channel info
        for i in range(1, self.n_channels + 1):
            parts = lines[i].split()
            self.channel_names.append(parts[8])  # Channel name
            self.gain.append(float(parts[2].split('/')[0]))  # ADC gain
            self.baseline.append(int(parts[4]))  # Baseline
            self.units.append(parts[2].split('/')[1])  # Units (e.g., mV)
        for line in lines[self.n_channels + 1:]:
            line = line.strip()

            if line.startswith("#Age:"):
                value = line.split(":", 1)[1].strip()
                self.age = None if value.lower() == "unknown" else int(value)

            elif line.startswith("#Sex:"):
                value = line.split(":", 1)[1].strip()
                self.sex = None if value.lower() == "unknown" else value

            elif line.startswith("#Dx:"):
                value = line.split(":", 1)[1].strip()
                self.dx_codes = [code.strip() for code in value.split(",")]
        
        return {
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'channel_names': self.channel_names,
            'units': self.units,
            'age': self.age,
            'sex': self.sex,
            'dx_codes': self.dx_codes
        }
    
    def read_data(self):
        """
        Read the .mat signal data and convert to physical units.
        
        Returns:
            np.ndarray: Signal array of shape (n_channels, n_samples).
        """
        mat_data = scipy.io.loadmat(self.data_path)
        
        if 'val' in mat_data:
            raw_signal = mat_data['val']
        else:
            # Find the first non-metadata key
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            raw_signal = mat_data[keys[0]]
        
        # Convert to physical units using gain and baseline
        self.signal = np.zeros_like(raw_signal, dtype=float)
        for i in range(self.n_channels):
            self.signal[i, :] = (raw_signal[i, :] - self.baseline[i]) / self.gain[i]
        
        return self.signal
    
    def read(self):
        """
        Read both header and signal data.
        
        Returns:
            ECGReader: Self with populated metadata and signal.
        """
        self.read_header()
        self.read_data()
        return self
    
    def get_time_array(self):
        """
        Generate time array in seconds.
        
        Returns:
            np.ndarray: Time values aligned with samples.
        """
        return np.arange(self.n_samples) / self.sampling_rate
    
    def plot(self, channels=None, start_time=0, duration=10):
        """
        Plot ECG signals
        
        Args:
            channels (list, optional): Channel indices to plot (None = all).
            start_time (float): Start time in seconds. Defaults to 0.
            duration (float): Duration to plot in seconds. Defaults to 10.
        
        Returns:
            matplotlib.figure.Figure: Figure handle.
        """
        if self.signal is None:
            raise ValueError("No data loaded. Call read() first.")
        
        if channels is None:
            channels = range(self.n_channels)
        
        # Convert time to samples
        start_sample = int(start_time * self.sampling_rate)
        end_sample = int((start_time + duration) * self.sampling_rate)
        end_sample = min(end_sample, self.n_samples)
        
        time = self.get_time_array()[start_sample:end_sample]
        
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, 2*len(channels)))
        if len(channels) == 1:
            axes = [axes]
        
        for idx, ch in enumerate(channels):
            axes[idx].plot(time, self.signal[ch, start_sample:end_sample])
            axes[idx].set_ylabel(f'{self.channel_names[ch]}\n({self.units[ch]})')
            axes[idx].grid(True, alpha=0.3)
            if idx == len(channels) - 1:
                axes[idx].set_xlabel('Time (s)')
        
        plt.suptitle(f'ECG Recording: {self.base_path.name}')
        plt.tight_layout()
        return fig

def find_ecg_records(root_dir):
    """
    Finds ECG records by locating .hea files
    Returns list of base paths (without suffix)
    """
    root_dir = Path(root_dir)
    hea_files = list(root_dir.rglob("*.hea"))
    return [f.with_suffix("") for f in hea_files]

class ECGDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): must contain 'filepath' and 't', 'y' columns
        """
        self.dataframe = dataframe.reset_index(drop=True)

        # Filter only valid ECG files
        def is_valid_ecg_file(path):
            path = Path(path)
            
            reader = ECGReader(path)
            reader.read_header()
            return True

        self.data = self.dataframe[self.dataframe['filepath'].apply(is_valid_ecg_file)].reset_index(drop=True)

    def __len__(self):
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load and return a single ECG sample.
        
        Args:
            idx (int): Sample index.
        
        Returns:
            dict: Batch with keys 'X', 't', 'y'.
        """
        path = self.data.iloc[idx]['filepath']
        treatment = torch.tensor(self.data.iloc[idx]['t'], dtype=torch.float32)
        outcome = torch.tensor(self.data.iloc[idx]['y'], dtype=torch.float32)

        # Load ECG
        reader = ECGReader(path)
        reader.read()  # reads header + mat
        waveform = reader.signal  # shape: (n_channels, n_samples)
        
        # Normalize
        waveform = (waveform - np.mean(waveform)) / np.std(waveform)

        waveform = torch.tensor(waveform, dtype=torch.float32)
        return {'X': waveform, 't': treatment, 'y': outcome}
