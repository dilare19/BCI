# online/lsl_receiver.py

import numpy as np
from pylsl import StreamInlet, resolve_stream

class LSLReceiver:

    def __init__(self,n_channels,fs,window_len):

        print("Looking for EEG stream...")

        streams = resolve_stream('type','EEG')

        self.inlet = StreamInlet(streams[0])

        print("EEG connected")

        self.n_channels = n_channels
        self.window_len = window_len

        self.buffer = np.zeros((n_channels,window_len))

    def update(self):

        sample,timestamp = self.inlet.pull_sample()

        sample = np.array(sample[:self.n_channels])

        self.buffer = np.roll(self.buffer,-1,axis=1)
        self.buffer[:,-1] = sample

        return self.buffer