from collections import Counter

import numpy as np
import pandas as pd

from core.Cache import Cache
from core.Request import RequestLoader


class CacheEnv:
    def __init__(self, capacity, request_path, top_k, time_slot_length):
        self.capacity = capacity
        self.loader = RequestLoader(request_path, time_slot_length)
        self.cache = Cache(capacity)
        self.top_k = top_k
    
    def reset(self):
        self.cache.clear()
        self.loader.reset()
        while not self.cache.is_full():
            request = self.loader.next()
            if not self.cache.find(request):
                self.cache.update(request)
    
    def step(self):
        # 获取一段时间内的请求
        requests = self.loader.next_time_slot()
        top_k_missed_videos, hit_mask = self._handle_requests(requests)
        
        # 打印当前缓存内容与候补
        print("Cache content: {}".format(self.cache.get_content()))
        print("Top K missed videos: {}".format(top_k_missed_videos))
        
        new_cache_content = set(self._get_new_cache_content(self.cache.get_content(), top_k_missed_videos))
        old_cache_content = set(self.cache.get_content())
        
        old_elements = old_cache_content.difference(new_cache_content)
        new_elements = new_cache_content.difference(old_cache_content)
        assert len(old_elements) == len(new_elements)
        for new_element, old_element in zip(new_elements, old_elements):
            self.cache.update(new_element, old_element)
        
        hit_rate = hit_mask.mean()
        
        return hit_rate
    
    def _handle_requests(self, requests):
        # 用于记录请求是否命中
        hit_mask = np.zeros_like(requests, dtype=np.bool)
        # 测试请求是否命中
        for idx, request in enumerate(requests):
            hit_mask[idx] = self.cache.find(request)
        
        # 选取未命中请求
        missed_requests = requests[~hit_mask]
        # 统计未命中请求内容的频率
        missed_frequencies = pd.DataFrame(Counter(missed_requests).items(),
                                          columns=['video_id', 'frequencies'])
        # 选取频率top k的视频id作为候补
        top_k_missed_videos = missed_frequencies.sort_values('frequencies', ascending=False).iloc[:self.top_k]
        top_k_missed_videos = np.array(top_k_missed_videos['video_id'])
        
        return top_k_missed_videos, hit_mask
    
    def _get_new_cache_content(self, cache_content, top_k_missed_videos):
        raise NotImplementedError
    
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
