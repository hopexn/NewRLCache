import numpy as np

from envs.CacheEnv import CacheEnv


class DdpgCacheEnv(CacheEnv):
    def __init__(self, capacity, request_path, top_k, time_slot_length):
        super().__init__(capacity, request_path, top_k, time_slot_length)
    
    def _get_new_cache_content(self, cache_content, top_k_missed_videos, hit_rate):
        # TODO
        candidates = np.concatenate([cache_content, top_k_missed_videos])
        return np.random.choice(candidates, self.capacity, replace=False)
