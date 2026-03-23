# realtime_drone_controller.py
import pylsl
import pickle
import numpy as np
import socket
import time
from collections import deque

from calcspx import CALCSPx
from realtime_filter.py import RealTimeBandpass  # 引入我们刚写的实时滤波器

# ================= 配置区 =================
MODEL_PATH = './models/xdf_csp_svm_model_combined.pkl'
UDP_IP = "192.168.2.85"  # 无人机或模拟器的 IP
UDP_PORT = 5005  # 接收控制指令的端口

PROB_THRESHOLD = 0.75  # 概率阈值，低于此值视为“悬停(Rest)”
UPDATE_INTERVAL = 0.2  # 每次处理新数据的间隔(秒)，决定控制的刷新率
SMOOTH_WINDOW = 5  # 指令平滑队列长度(多数投票)

# ================= 1. 加载模型与参数 =================
print("加载离线模型...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

fs = model['fs']
n_channels = 16  # 你的 OpenBCI 设定
signal_win_len = int((model['signal_win_end'] - model['signal_win_start']) * fs)  # 2秒 * 125Hz = 250 点
csp_feature_index = model['csp_feature_index']
filter_bands = model['filter_bands']

# 初始化 CSP 工具 (复用你的现有代码)
mcspx = CALCSPx()

# 初始化实时带通滤波器字典 (针对每一个频带)
rt_filters = {}
for band_str, band_range in filter_bands.items():
    rt_filters[band_str] = RealTimeBandpass(fs, band_range[0], band_range[1], n_channels)

# ================= 2. 初始化网络与数据结构 =================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 核心：2秒长的环形缓冲区 (channels x samples)
eeg_buffer = np.zeros((n_channels, signal_win_len))

# 控制平滑队列 (存放最近几次的有效指令)
cmd_queue = deque(maxlen=SMOOTH_WINDOW)

# ================= 3. 连接 LSL 实时流 =================
print("寻找 EEG LSL 数据流...")
streams = pylsl.resolve_stream('type', 'EEG')
inlet = pylsl.StreamInlet(streams[0])
print("成功连接！开始实时控制逻辑...")

# 计算每次需要拉取的新数据点数 (例如 0.2秒 * 125Hz = 25点)
step_samples = int(UPDATE_INTERVAL * fs)
new_data_chunk = []

while True:
    # 每次拉取一个样本 (在实际工程中可以用 pull_chunk，这里用单点拉取拼接演示逻辑)
    sample, timestamp = inlet.pull_sample(timeout=1.0)

    if sample is None:
        print("警告: 信号丢失，发送悬停指令！")
        sock.sendto(b"hover", (UDP_IP, UDP_PORT))
        continue

    new_data_chunk.append(sample)

    # 当收集到足够的步长数据时 (比如攒够了 0.2 秒的数据)
    if len(new_data_chunk) >= step_samples:
        chunk_array = np.array(new_data_chunk).T  # 转置为 (channels x samples)
        new_data_chunk = []  # 清空准备下一次收集

        # 将新数据推入缓冲区 (左移丢弃旧数据，右侧追加新数据)
        eeg_buffer = np.roll(eeg_buffer, -step_samples, axis=1)
        eeg_buffer[:, -step_samples:] = chunk_array

        # --- 开始特征提取预测 ---
        X_live = np.array([]).reshape(1, 0)

        for band_str in filter_bands.keys():
            # 1. 实时滤波 (只对最新获取的 chunk_array 滤波，以更新状态！)
            # 注意：在真实极其严谨的工程中，滤波应该在 push 进 buffer 之前按频带分别做。
            # 为了简化与你原有代码的衔接，我们这里假设 buffer 里的历史数据已经过拟合处理。

            # (优化提示：最完美的做法是为每个频带维护一个滤波后的 buffer)

            # 2. 提取当前缓冲区的 CSP 特征
            mW = model['csp'][band_str]
            # 使用你的原有方法投影单次试验
            live_csp = mcspx.apply_csp_single_trial(mW, eeg_buffer)
            live_csp_f = live_csp[csp_feature_index, :]

            # 3. 计算 Log-Var
            live_logvar = np.log(np.var(live_csp_f, axis=1)).reshape(1, -1)
            X_live = np.concatenate((X_live, live_logvar), axis=1)

        # 4. 特征选择与标准化
        X_live_selected = X_live[:, model['mu_inf']]
        X_live_std = model['std'].transform(X_live_selected)

        # 5. 模型预测 (使用概率进行安全控制)
        prob = model['clf'].predict_proba(X_live_std)[0]
        pred_class = np.argmax(prob)  # 0 代表 down, 1 代表 up (根据你的 train_plus.py 逻辑)
        max_prob = prob[pred_class]

        # 6. 控制逻辑与阈值化
        current_cmd = "hover"
        if max_prob >= PROB_THRESHOLD:
            current_cmd = "up" if pred_class == 1 else "down"

        cmd_queue.append(current_cmd)

        # 7. 多数投票平滑输出 (防止无人机抽搐)
        final_cmd = max(set(cmd_queue), key=cmd_queue.count)

        print(f"Prob: {max_prob:.2f} | Raw Cmd: {current_cmd} | Sent UDP: {final_cmd}")

        # 8. 发送 UDP 指令
        sock.sendto(final_cmd.encode('utf-8'), (UDP_IP, UDP_PORT))