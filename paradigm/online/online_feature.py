# online/online_feature.py

import numpy as np
from bandpassx import BANDPASSx
from calcspx import CALCSPx

class OnlineFeature:

    def __init__(self,model):

        self.csp_all = model['csp']
        self.mu_inf = model['mu_inf']
        self.filter_bands = model['filter_bands']
        self.csp_feature_index = model['csp_feature_index']
        self.std = model['std']
        self.fs = model['fs']

        self.mcspx = CALCSPx()

    def extract(self,eeg_window):

        X = []

        for band_key,band_range in self.filter_bands.items():

            lo,hi = band_range

            mband = BANDPASSx(self.fs,lo,hi)

            tmp = eeg_window[:,:,None]

            filtered = mband.apply_filter(tmp)

            W = self.csp_all[band_key]

            trials_csp = self.mcspx.apply_csp(W,filtered)

            trials_csp_f = trials_csp[self.csp_feature_index,:,:]

            feature = np.log(np.var(trials_csp_f,axis=1))

            X.extend(feature.flatten())

        X = np.array(X)

        X = X[self.mu_inf]

        X = X.reshape(1,-1)

        X = self.std.transform(X)

        return X