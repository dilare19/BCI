# -*- coding: utf-8 -*-
import numpy as np
import pickle
import pyxdf
import time
import json
import socket  # 导入套接字库
from collections import deque  # 新增：用于投票缓冲区
from bandpassx import BANDPASSx as bandpassx
from calcspx import CALCSPx as calcspx

# ==== 1. 加载模型与配置 ====
model_path = './models/xdf_csp_svm_model_combined.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

clf = model_data['clf']
csp_all = model_data['csp']
mu_inf_rank = model_data['mu_inf']
filter_bands = model_data['filter_bands']
fs = model_data['fs']
std_scaler = model_data['std']
csp_feat_idx = model_data['csp_feature_index']
win_samples = int((model_data['signal_win_end'] - model_data['signal_win_start']) * fs)

# ==== 2. 加载 XDF 数据并提取 Ground Truth ====
# 请确保路径正确
xdf_path = r'C:\Users\32434\Documents\CurrentStudy\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
streams, _ = pyxdf.load_xdf(xdf_path)

eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]
marker_stream = [s for s in streams if s['info']['type'][0] == 'Markers'][0]

eeg_data = np.array(eeg_stream['time_series']).T[:16, :]
eeg_times = np.array(eeg_stream['time_stamps'])
marker_times = np.array(marker_stream['time_stamps'])
marker_values = [m[0] for m in marker_stream['time_series']]

ground_truth = np.full(eeg_data.shape[1], -1)
for i, val in enumerate(marker_values):
    if val in [11, 12]:
        start_idx = np.searchsorted(eeg_times, marker_times[i])
        end_idx = start_idx + int(2.5 * fs)
        label = 1 if val == 11 else 0
        ground_truth[start_idx: end_idx] = label


# ==== 3. 模拟实时处理与验证 ====
def send_port(args):
    pass


def run_validation():
    # --- A. 初始化 UDP 发送端 ---
    target_ip = "192.168.2.85"  # ROS 电脑 IP
    send_port = 5005  # 发送给电脑2的端口
    local_recv_port = 5007
    # udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 【关键修改】绑定本地端口，这样 5007 就成了你的固定接收端口
    try:
        # '' 代表监听本机所有网卡 IP 上的 5007 端口
        udp_socket.bind(('', local_recv_port))
        print(f"成功绑定本地端口: {local_recv_port}，准备接收回传...")
    except Exception as e:
        print(f"绑定端口 {local_recv_port} 失败: {e}")
        return

    udp_socket.settimeout(0.1)  # 设置 100ms 超时
    print(f"UDP 已启动，目标: {target_ip}:{send_port}")

    mcspx = calcspx()
    step_size = int(fs * 0.5)

    # --- 核心改进：改为概率平滑缓冲区 ---
    smooth_win_len = 5  # 平滑窗口长度，建议 4-6
    prob_buffer = deque(maxlen=smooth_win_len)

    # 依然保留置信度阈值，作为进入缓冲区的门槛
    confidence_threshold = 0.65  # 使用平滑后，单次阈值可以稍微调低一点

    # 统计指标
    correct_count = 0
    total_decisions = 0
    skipped_count = 0
    # # --- 优化参数设置 ---
    # confidence_threshold = 0.75  # 置信度阈值：只有高于75%的预测才会被采纳
    # vote_len = 3  # 投票箱长度：参考最近3次有效预测
    # prediction_buffer = deque(maxlen=vote_len)

    # # 统计指标
    # correct_count = 0  # 最终决策正确的次数
    # total_decisions = 0  # 总共做出的有效决策次数
    # skipped_count = 0  # 因为置信度不足被过滤的次数

    print(f"{'时间(s)':<10} | {'原始预测':<8} | {'最终决策':<8} | {'真实值':<8} | {'判定':<6} | {'置信度'}")
    print("-" * 85)

    for start_idx in range(0, eeg_data.shape[1] - win_samples, step_size):
        actual_label = ground_truth[start_idx + win_samples]

        # 1. 核心修复：在此处初始化所有打印变量的默认值
        timestamp_in_xdf = start_idx / fs
        final_decision_str = "无"
        latency_str = "--"
        res_str = "--"
        raw_confidence = 0.0
        raw_pred = -1

        if actual_label == -1:
            prob_buffer.clear()
            # 如果不在任务区，也打印一行，或者直接 continue
            # print(f"{timestamp_in_xdf:<10.2f} | 休息中...")
            continue

        # --- 特征提取流程 (保持不变) ---
        current_win = eeg_data[:, start_idx: start_idx + win_samples]
        X_live_list = []
        for band_key, band_range in filter_bands.items():
            bp = bandpassx(fs, lo=band_range[0], hi=band_range[1])
            f_win = bp.apply_filter_2d(current_win)
            mW = csp_all[band_key]
            csp_f = mcspx.apply_csp_single_trial(mW, f_win)[csp_feat_idx, :]
            X_live_list.append(np.log(np.var(csp_f, axis=1)))

        X_std = std_scaler.transform(np.concatenate(X_live_list).reshape(1, -1)[:, mu_inf_rank])

        # --- 预测 ---
        probs = clf.predict_proba(X_std)[0]
        raw_pred = int(np.argmax(probs))
        raw_confidence = np.max(probs)

        # 2. 逻辑判断
        if raw_confidence >= 0.60:
            prob_buffer.append(probs)

            if len(prob_buffer) == smooth_win_len:
                avg_probs = np.mean(prob_buffer, axis=0)
                final_pred = np.argmax(avg_probs)
                smoothed_conf = avg_probs[final_pred]

                if smoothed_conf >= 0.70:
                    total_decisions += 1
                    final_decision_str = "向上" if final_pred == 1 else "向下"

                    # 判断正确性用于统计
                    if final_pred == actual_label:
                        correct_count += 1
                        res_str = "√"
                    else:
                        res_str = "×"

                    # --- UDP 发送与延迟测量 ---
                    msg = f"{final_pred},{time.time()}"
                    t_start = time.perf_counter()
                    try:
                        # 发送到电脑 2 的 5005 端口
                        udp_socket.sendto(msg.encode('utf-8'), (target_ip, send_port))

                        # 接收回传（此时 socket 已经在 5007 监听）
                        echo_data, addr = udp_socket.recvfrom(1024)

                        latency_ms = (time.perf_counter() - t_start) * 1000
                        latency_str = f"{latency_ms:.2f}ms"
                    except socket.timeout:
                        latency_str = "TIMEOUT"
                    except Exception as e:
                        latency_str = "ERROR"
        else:
            skipped_count += 1
            final_decision_str = "过滤"

        # 3. 最终打印：此时 latency_str 等变量无论如何都有值了
        raw_name = "向上" if raw_pred == 1 else "向下"
        true_name = "向上" if actual_label == 1 else "向下"

        print(f"{timestamp_in_xdf:<10.2f} | {raw_name:<8} | {final_decision_str:<8} | "
              f"{true_name:<8} | {res_str:<6} | {raw_confidence:.2f} | {latency_str}")

        # --- 总结报告 ---
    udp_socket.close()  # 结束后关闭连接
    print("-" * 85)
    if total_decisions > 0:
        accuracy = correct_count / total_decisions
        print(f"模拟结束报告：")
        print(f"1. 有效决策总数: {total_decisions} (通过置信度过滤并完成投票的次数)")
        print(f"2. 被过滤的模糊窗口数: {skipped_count}")
        print(f"3. 最终决策准确率 (平滑后): {accuracy:.2%}")
        print(f"\n提示：如果决策准确率大幅提升，说明平滑机制有效；如果决策总数太少，请调低 confidence_threshold。")
    else:
        print("未做出有效决策。请检查数据 Marker 或尝试调低 confidence_threshold。")


if __name__ == '__main__':
    run_validation()