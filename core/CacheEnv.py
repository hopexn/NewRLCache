from collections import Counter

import numpy as np
import pandas as pd

from agents.Agent import Agent
from core.Cache import Cache
from core.Request import RequestLoader


class CacheEnv:
    def __init__(self, capacity, loader: RequestLoader, top_k, agent: Agent):
        self.capacity = capacity
        self.loader = loader
        self.cache = Cache(capacity)
        self.top_k = top_k
        self.agent = agent
    
    def warm_up(self):
        while not self.cache.full():
            requests = self.loader.next()
            for request in requests:
                if self.cache.full():
                    break
                if self.cache.find(request) == -1:
                    position = len(self.cache)
                    self.cache.put(position, request)
    
    def step(self):
        # 获取一段时间内的请求
        requests = self.loader.next()
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
        # 打印当前缓存内容与候补
        print("Cache content: {}".format(self.cache.entries))
        print("Candidates: {}".format(top_k_video))
        
        self.agent.forward()
        self.agent.backward()
        
        
        hit_rate = np.mean(hit_mask)
        return hit_rate
    
    
    def feature_extract(self):
    