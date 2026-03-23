# realtime_filter.py
import numpy as np
import scipy.signal


class RealTimeBandpass:
    def __init__(self, fs, lo, hi, nchannels):
        """
        初始化因果滤波器并保留状态矩阵 (zi)
        """
        self.fs = fs
        self.nchannels = nchannels
        # 依然使用 iirfilter，但准备用于前向滤波
        self.b, self.a = scipy.signal.iirfilter(6, [lo / (fs / 2.0), hi / (fs / 2.0)])

        # 计算单通道的初始状态 zi
        zi_1d = scipy.signal.lfilter_zi(self.b, self.a)

        # 为所有通道复制状态，形状为 (nchannels, zi_length)
        self.zi = np.array([zi_1d] * nchannels)

    def filter_chunk(self, chunk):
        """
        传入最新的数据块 (channels x samples)，返回滤波后的数据块
        """
        filtered_chunk = np.zeros_like(chunk)
        # 对每个通道单独进行滤波，并更新该通道的 zi 状态
        for ch in range(self.nchannels):
            filtered_chunk[ch, :], self.zi[ch, :] = scipy.signal.lfilter(
                self.b, self.a, chunk[ch, :], zi=self.zi[ch, :]
            )
        return filtered_chunk