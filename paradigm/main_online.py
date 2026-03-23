# main_online.py

import pickle
import time
from collections import deque
import numpy as np

from online.lsl_receiver import LSLReceiver
from online.online_feature import OnlineFeature
from online.online_predict import OnlinePredict
from online.drone_controller import DroneController


####################################
# 1 读取模型
####################################

model = pickle.load(open('./models/xdf_csp_svm_model003.pkl','rb'))

fs = model['fs']
window_start = model['signal_win_start']
window_end = model['signal_win_end']

window_len = int((window_end-window_start)*fs)

n_channels = 16

####################################
# 2 初始化模块
####################################

receiver = LSLReceiver(n_channels,fs,window_len)

feature = OnlineFeature(model)

predictor = OnlinePredict(model)

drone = DroneController()

####################################
# 3 控制参数
####################################

threshold = 0.75
step_time = 0.05          # 每次预测间隔
stability_window = 3      # 连续次数判断
confidence_queue_len = 5  # 滑动平均窗口
pred_queue = deque(maxlen=stability_window)
prob_queue = deque(maxlen=confidence_queue_len)
####################################
# 4 主循环
####################################

while True:

    eeg_window = receiver.update()

    # 检查 buffer 是否填满
    if np.any(eeg_window[:, 0] == 0):
        continue

    # baseline / z-score 校正
    eeg_window = eeg_window - np.mean(eeg_window, axis=1, keepdims=True)
    # eeg_window = eeg_window / (np.std(eeg_window, axis=1, keepdims=True) + 1e-10)

    X = feature.extract(eeg_window)

    pred,confidence = predictor.predict(X)

    # 滑动平均平滑
    prob_queue.append(confidence)
    avg_confidence = np.mean(prob_queue)

    print(f"Pred:{pred} Confidence:{confidence:.3f} Avg:{avg_confidence:.3f}")

    # 阈值判断 + 稳定器
    if avg_confidence >= threshold:
        pred_queue.append(pred)
    else:
        pred_queue.append(None)

    if len(pred_queue) == stability_window and all(p == pred_queue[0] and p is not None for p in pred_queue):
        stable_pred = pred_queue[0]

        # 执行无人机动作
        if stable_pred == 1:

            drone.up()

        else:

            drone.down()

        # 清空队列，避免重复执行
        pred_queue.clear()

    time.sleep(step_time)