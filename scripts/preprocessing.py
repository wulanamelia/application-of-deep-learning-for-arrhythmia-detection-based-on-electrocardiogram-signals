import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import pickle

# Lokasi dataset
DATASET_PATH = './dataset/'

# Buat direktori output
os.makedirs('output', exist_ok=True)

# Filter bandpass 0.5-40 Hz
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

# Loop semua file
X = []
y = []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith('.dat'):
        record_name = filename.replace('.dat', '')
        record = wfdb.rdrecord(DATASET_PATH + record_name)
        signal = record.p_signal[:,0]  # Ambil channel pertama
        filtered = bandpass_filter(signal)

        # Normalisasi
        scaler = StandardScaler()
        norm_signal = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()

        # Segmentasi contoh (window 180)
        for i in range(0, len(norm_signal) - 180, 180):
            segment = norm_signal[i:i+180]
            X.append(segment)
            # Dummy label: contoh binary (0 normal, 1 aritmia)
            y.append(0 if int(record_name) < 105 else 1)

print(f'Total segmen: {len(X)}')

X = np.array(X)
y = np.array(y)

np.save('output/X.npy', X)
np.save('output/y.npy', y)

print('Selesai simpan data preprocessing ke output/')
