import pyxdf
import mne
import numpy as np

streams, header = pyxdf.load_xdf(
    r"C:\Users\32434\Documents\CurrentStudy\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
)
print(len(streams))
print(streams[0].keys())

for i, s in enumerate(streams):
    print(i, s['info']['name'], s['info']['type'])

eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]

eeg_data = eeg_stream['time_series']   # shape: (n_samples, n_channels)
eeg_time = eeg_stream['time_stamps']
fs = float(eeg_stream['info']['nominal_srate'][0])

print(eeg_data.shape, fs)

marker_stream = [s for s in streams if s['info']['type'][0] == 'Markers'][0]

markers = marker_stream['time_series']
marker_time = marker_stream['time_stamps']

for t, m in zip(marker_time, markers):
    print(t, m)

info = mne.create_info(
    ch_names=[f'EEG{i}' for i in range(eeg_data.shape[1])],
    sfreq=fs,
    ch_types='eeg'
)

raw = mne.io.RawArray(eeg_data.T, info)
raw.plot()