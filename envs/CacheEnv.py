from collections import Counter

import numpy as np
import pandas as pd

from core.Cache import Cache
from core.Request import RequestLoader


class CacheEnv:
    def __init__(self, capacity, loader: RequestLoader, top_k):
        self.capacity = capacity
        self.loader = loader
        self.cache = Cache(capacity)
        self.top_k = top_k
    
    def reset(self):
        while not self.cache.full():
            request = self.loader.next()
            if self.cache.find(request) == -1:
                position = len(self.cache)
                self.cache.put(position, request)
        
        requests = self.loader.next_time_slot()
        top_k_missed_videos, hit_mask = self._handle_requests(requests)
        observation = np.concatenate([self.cache.entries, top_k_missed_videos])
        hit_rate = np.mean(hit_mask)
        return observation, hit_rate
    
    def step(self):
        # 获取一段时间内的请求
        requests = self.loader.next_time_slot()
        top_k_missed_videos, hit_rate = self._handle_requests(requests)
        
        # 打印当前缓存内容与候补
        print("Cache content: {}".format(self.cache.entries))
        print("Candidates: {}".format(top_k_missed_videos))
        
        observation = np.concatenate([self.cache.entries, top_k_missed_videos])
        
        return hit_rate
    
    def _handle_requests(self, requests):
        # 用于记录请求是否命中
        hit_mask = np.zeros_like(requests, dtype=np.bool)
        # 测试请求是否命中
        for idx, request in enumerate(requests):
            position = self.cache.find(request)
            hit_mask[idx] = (position != -1)
        
        # 选取未命中请求
        missed_requests = requests[~hit_mask]
        # 统计未命中请求内容的频率
        missed_frequencies = pd.DataFrame(Counter(missed_requests).items(),
                                          columns=['video_id', 'frequencies'])
        # 选取频率top k的视频id作为候补
        top_k_video = missed_frequencies.sort_values('frequencies', ascending=False).iloc[:self.top_k]
        top_k_video = np.array(top_k_video['video_id'])
        
        return top_k_video, hit_mask
    
    def before_step(self):
        """
        每一步开始执行前的操作写在这
        """
        pass
    
    def after_step(self):
        """
        每一步结束后的操作写在这
        """
        pass
    
    def before_game(self):
        """
        游戏开始前的操作写在这
        """
        pass
    
    def after_game(self):
        """
        游戏结束后的操作写在这
        """
        pass
