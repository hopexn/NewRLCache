import os

import numpy as np
import pandas as pd
from numba import prange, jit


@jit(nopython=True)
def _get_historical_frequency(
        frequencies,
        historical_requests,
        videos,
        historical_pointers,  # historical_pointers的数量应该比num_time_steps多1
        num_time_steps,
        time_step,
):
    videos = videos.reshape(-1, 1)
    
    for i in prange(num_time_steps):
        ptr_begin = historical_pointers[i]
        ptr_end = historical_pointers[i + 1]
        
        if ptr_begin == ptr_end:
            continue
        
        requests = historical_requests[ptr_begin:ptr_end].reshape(1, -1)
        masks = np.equal(videos, requests)
        
        frequencies[:, i] = np.sum(masks, axis=1)
    
    # frequencies = (frequencies >= 1)
    frequencies = frequencies / time_step
    
    return frequencies


class RequestLoader:
    def __init__(self, path, time_slot_length):
        self.path = path
        self.time_slot_length = time_slot_length
        
        # 文件不存在则报错
        if not os.path.exists(path):
            print("File Not Found: {}".format(path))
            raise FileNotFoundError
        
        requests = pd.read_csv(path)
        self.video_ids = np.array(requests['video_id'].astype(np.int))
        self.timestamps = np.array(requests['timestamp'].astype(np.int))
        
        self.ptr = 0  # 指向下一个时间片请求序列的开始
        self.nb_requests = len(requests)  # 序列总数
    
    def next_time_slot(self):
        ptr_begin = self.ptr
        time_slot_begin = self.timestamps[ptr_begin]
        time_slot_end = time_slot_begin + self.time_slot_length
        
        while self.timestamps[self.ptr] < time_slot_end \
                and not self.finished():
            self.ptr += 1
        
        ptr_end = self.ptr
        
        requests_slice = self.video_ids[ptr_begin:ptr_end]
        return requests_slice
    
    def next(self):
        if not self.finished():
            request = self.video_ids[self.ptr]
        else:
            request = -1
        
        self.ptr += 1
        
        return request
    
    def finished(self):
        return self.ptr >= self.nb_requests
    
    def reset(self):
        self.ptr = 0
    
    def __len__(self):
        return self.nb_requests
    
    def get_last_access_time(self, videos: np.array):
        last_access_times = np.zeros_like(videos) - 1
        for idx, video in enumerate(videos):
            tmp_ptr = self.ptr - 1
            last_access_time = -1
            while tmp_ptr >= 0:
                if self.video_ids[tmp_ptr] == video:
                    last_access_time = self.timestamps[tmp_ptr]
                    break
                tmp_ptr -= 1
            last_access_times[idx] = last_access_time
        
        return last_access_times
    
    def get_frequencies(self, videos, history_window_length=2000):
        history = np.zeros((history_window_length,)) - 1
        history[max(0, history_window_length - self.ptr):] = self.video_ids[
                                                             max(0, self.ptr - history_window_length): self.ptr]
        videos = np.expand_dims(videos, axis=1)
        history = np.expand_dims(history, axis=0)
        frequencies = np.mean(videos == history, axis=1, keepdims=False)
        return frequencies
    
    def get_frequencies2(self, videos, history_window_length_list: list):
        max_length = np.max(history_window_length_list)
        num_window_length = len(history_window_length_list)
        
        history = np.zeros((max_length,)) - 1
        history[max(0, max_length - self.ptr):] = self.video_ids[
                                                  max(0, self.ptr - max_length): self.ptr]
        videos = np.expand_dims(videos, axis=1)
        history = np.expand_dims(history, axis=0)
        hit_mask = videos == history
        frequencies = np.zeros((len(videos), num_window_length))
        for idx, window_length in enumerate(history_window_length_list):
            frequencies[:, idx] = np.mean(hit_mask[:, -window_length:], axis=1)
        
        return frequencies
