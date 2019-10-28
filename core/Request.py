import os

import numpy as np
import pandas as pd


class RequestLoader:
    def __init__(self, path, time_slot_length):
        self.path = path
        self.time_slot_length = time_slot_length
        
        # 文件不存在则报错
        if not os.path.exists(path):
            print("File Not Found: {}".format(path))
            raise FileNotFoundError
        
        self.requests = pd.read_csv(path)
        self.requests['timestamp'] = self.requests['timestamp'].astype(np.int)
        self.requests['video_id'] = self.requests['video_id'].astype(np.int)
        
        self.ptr = 0  # 指向下一个时间片请求序列的开始
        self.nb_requests = len(self.requests)  # 序列总数
    
    def next_time_slot(self):
        ptr_begin = self.ptr
        time_slot_begin = self.requests.loc[ptr_begin, 'timestamp']
        time_slot_end = time_slot_begin + self.time_slot_length
        
        while self.requests.loc[self.ptr, 'timestamp'] < time_slot_end \
                and not self.finished():
            self.ptr += 1
        
        ptr_end = self.ptr
        
        requests_slice = self.requests.loc[ptr_begin:ptr_end - 1, 'video_id']
        return np.array(requests_slice)
    
    def next(self):
        if not self.finished():
            request = self.requests.loc[self.ptr, 'video_id']
        else:
            request = -1
        return request
    
    def finished(self):
        return self.ptr >= self.nb_requests
    
    def reset(self):
        self.ptr = 0
    
    def __len__(self):
        return self.nb_requests
