# mock_lsl_streamer.py
import pyxdf
import pylsl
import time
import numpy as np

# ==== 配置区 ====
# 填入你用于测试的离线数据路径 (借用你 train_plus.py 中的文件)
XDF_FILE = r'C:\Users\32434\Documents\CurrentStudy\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
TARGET_FS = 125.0  # OpenBCI 的采样率


def run_mock_stream():
    print(f"正在加载数据: {XDF_FILE}...")
    streams, header = pyxdf.load_xdf(XDF_FILE)

    eeg_data = None
    for stream in streams:
        # 寻找 EEG 数据流 (根据你的数据类型可能叫 'EEG' 或 'openbci_eeg' 等)
        if stream['info']['type'][0].lower() == 'eeg':
            eeg_data = stream['time_series']
            n_channels = int(stream['info']['channel_count'][0])
            break

    if eeg_data is None:
        print("错误: 在 XDF 文件中找不到 EEG 流！")
        return

    print(f"提取到 {n_channels} 通道数据，共 {len(eeg_data)} 个采样点。")

    # 1. 创建 LSL Stream Info (这会让你的预测脚本找到它)
    info = pylsl.StreamInfo(
        name='OpenBCI_Mock',
        type='EEG',
        channel_count=n_channels,
        nominal_srate=TARGET_FS,
        channel_format='float32',
        source_id='mock_eeg_001'
    )

    # 2. 创建 Outlet (出口)
    outlet = pylsl.StreamOutlet(info)
    print("模拟 LSL 流已启动！现在可以运行你的实时控制脚本了。")
    print("按 Ctrl+C 停止推流。")

    # 3. 模拟实时推流循环
    try:
        # 计算每个样本之间的理论等待时间
        sleep_time = 1.0 / TARGET_FS

        for i in range(len(eeg_data)):
            start_time = time.perf_counter()

            # 推送一个采样点
            sample = eeg_data[i, :].tolist()
            outlet.push_sample(sample)

            # 补偿代码执行消耗的时间，实现精准的 125Hz 推流
            elapsed = time.perf_counter() - start_time
            if elapsed < sleep_time:
                time.sleep(sleep_time - elapsed)

            if i % int(TARGET_FS * 5) == 0:  # 每 5 秒打印一次进度
                print(f"已播放 {i / TARGET_FS:.1f} 秒的数据...")

    except KeyboardInterrupt:
        print("\n模拟推流已手动停止。")


if __name__ == '__main__':
    run_mock_stream()