import numpy as np

from envs.CacheEnv import CacheEnv


class LruCacheEnv(CacheEnv):
    def __init__(self, capacity, request_path, top_k, time_slot_length):
        super().__init__(capacity, request_path, top_k, time_slot_length)
    
    def _get_new_cache_content(self, cache_content, top_k_missed_videos):
        candidates = np.concatenate([cache_content, top_k_missed_videos])
        last_access_time = self.loader.get_last_access_time(candidates)
        return candidates[np.argpartition(last_access_time, -self.capacity)[-self.capacity:]]
