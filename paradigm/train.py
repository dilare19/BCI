# -*- coding: utf-8 -*-
import pyxdf
import numpy as np
import mne
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import feature_selection

import os
import json

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline

with open('task_markers.json', 'r') as f:
    marker_map = json.load(f)
# ==== 用户配置 ====
xdf_file = r'C:\Users\32434\Documents\CurrentStudy\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
# training_data_folder = './training_data_curctrl/'  # 可选离线训练数据
model_save_path = './models/xdf_csp_svm_model003.pkl'

# EEG 配置
fs = 125  # OpenBCI 常用  采样率
n_channels = 16
trial_start_marker = marker_map['curctrl_up'][0]
trial_end_marker = marker_map['curctrl_down'][0]

# CSP 配置
csp_feature_index = np.array([0,1,2,3,-4,-3,-2,-1])
k_top_features = 6  # mutual info 选取前 k 个特征

# 带通滤波段
band_interval = 4
filter_band = np.arange(4, 24, step=int(band_interval/2))  # 4-24 Hz, 50% overlap
filter_bands_str_num = {f"{lo:02d}_{lo+band_interval:02d}":[lo, lo+band_interval] for lo in filter_band}

# ==== 读取 XDF 文件 ====
streams, header = pyxdf.load_xdf(xdf_file)
# 假设 EEG 流为第一个，Markers 流为第二个
eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]
marker_stream = [s for s in streams if s['info']['type'][0] == 'Markers'][0]

eeg_data = np.array(eeg_stream['time_series']).T  # shape: n_channels x n_samples

# === 调试：检查离线数据量级 ===
print("--- 离线数据 (train.py) 量级检查 ---")
print(f"数据形状: {eeg_data.shape}")
print(f"前5个采样点示例 (第1通道): {eeg_data[0, :5]}")
print(f"最大值: {np.max(eeg_data):.2f}")
print(f"最小值: {np.min(eeg_data):.2f}")
print(f"平均值: {np.mean(eeg_data):.2f}")
print("----------------------------------")

eeg_times = np.array(eeg_stream['time_stamps'])
marker_times = np.array(marker_stream['time_stamps'])
marker_values = [m[0] for m in marker_stream['time_series']]

# ==== 生成 MNE Raw 对象 ====
info = mne.create_info(ch_names=[f"Ch{i+1}" for i in range(n_channels)],
                       sfreq=fs, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')

# ==== 构建事件矩阵 ====
# MNE 事件格式: n_events x 3, [sample, 0, event_code]
events = []
event_dict = {}
for i, marker in enumerate(marker_values):
    sample_idx = np.searchsorted(eeg_times, marker_times[i])
    event_code = int(marker)
    events.append([sample_idx, 0, event_code])
    #event_dict[marker] = event_code
    event_dict[event_code] = event_code
events = np.array(events)

# ==== 选择 trial ====
labels = [event_dict[trial_start_marker], event_dict[trial_end_marker]]
events_marker = events[np.isin(events[:,2], labels)]

cl_lab = ['up','down']
evnet_dict_marker = {
    'up': event_dict[trial_start_marker],
    'down': event_dict[trial_end_marker]
}

# ==== Epochs ====
signal_win_start = 0.5
signal_win_end = 2.5  # 秒
epochs = mne.Epochs(raw, events_marker, event_id=evnet_dict_marker,
                    tmin=signal_win_start, tmax=signal_win_end, baseline=None, preload=True)

# ==== 构建 trial 数据 ====
trials_raw = {}
trial_num = {}
for cl in cl_lab:
    tmp = epochs[cl].get_data()  # shape: n_epochs x n_channels x n_samples
    trials_raw[cl] = tmp.transpose((1, 2, 0))  # shape: n_channels x n_samples x n_trials
    trial_num[cl] = trials_raw[cl].shape[2]

# ==== 带通滤波 ====
from bandpassx import BANDPASSx as bandpassx
trials_all = {}
for band_str, band_range in filter_bands_str_num.items():
    lo, hi = band_range
    mbandpass = bandpassx(fs, lo, hi)
    tmp = {}
    for cl in cl_lab:
        tmp[cl] = mbandpass.apply_filter(trials_raw[cl])
    trials_all[band_str] = tmp

# ==== CSP + logvar 特征 ====
from calcspx import CALCSPx as calcspx
mcspx = calcspx()
X = np.array([]).reshape(trial_num['up']+trial_num['down'], 0)

csp_all = {}
for band_key, trials in trials_all.items():
    mW = mcspx.get_csp_w(trials['up'], trials['down'])
    csp_all[band_key] = mW
    trials_csp_up = mcspx.apply_csp(mW, trials['up'])
    trials_csp_down = mcspx.apply_csp(mW, trials['down'])

    trials_csp_f = {
        'up': trials_csp_up[csp_feature_index,:,:],
        'down': trials_csp_down[csp_feature_index,:,:]
    }

    trials_logvar = {
        'up': np.log(np.var(trials_csp_f['up'], axis=1)),
        'down': np.log(np.var(trials_csp_f['down'], axis=1))
    }

    x = np.concatenate((trials_logvar['up'].T, trials_logvar['down'].T), axis=0)
    X = np.concatenate((X, x), axis=1)

y = np.concatenate((np.ones(trial_num['up']), np.zeros(trial_num['down'])), axis=0)

# ==== 特征选择 ====
mutual_info = feature_selection.mutual_info_classif(X, y)
mutual_info_rank_use = np.argsort(mutual_info)[::-1][:k_top_features]
X_train = X[:, mutual_info_rank_use]

# ==== 标准化 + SVM 训练 ====
std = StandardScaler()
X_train_std = std.fit_transform(X_train)

params = {
    'C':[0.1,1,10],
    'gamma':['scale', 0.1, 0.01],
    'kernel':['rbf','linear']
}
grid = GridSearchCV(SVC(probability=True), param_grid=params, cv=5)
grid.fit(X_train_std, y)

print("Best parameters:", grid.best_params_)
print("Cross-validation accuracy: %.4f" % grid.best_score_)

clf = SVC(C=grid.best_params_['C'],
          gamma=grid.best_params_['gamma'],
          kernel=grid.best_params_['kernel'],
          probability=True)
clf.fit(X_train_std, y)

# 预测（注意：这是训练集预测，只用于展示指标）
y_pred = clf.predict(X_train_std)
y_prob = clf.predict_proba(X_train_std)[:,1]

print("\n========== Classification Report =========")
print(classification_report(y, y_pred, target_names=['down','up']))

print("\n=========== Confusion Matrix ===========")
print(confusion_matrix(y, y_pred))

print("\n================== AUC ====================")
print("AUC: %.4f" % roc_auc_score(y, y_prob))

# ==== 保存模型 ====
train_model = {
    'clf': clf,
    'csp': csp_all,
    'mu_inf': mutual_info_rank_use,
    'filter_bands': filter_bands_str_num,
    'fs': fs,
    'signal_win_start': signal_win_start,
    'signal_win_end': signal_win_end,
    'csp_feature_index': csp_feature_index,
    'std': std
}

if not os.path.exists('./models'):
    os.makedirs('./models')
pickle.dump(train_model, open(model_save_path, 'wb'))

print("模型训练完成并保存到:", model_save_path)