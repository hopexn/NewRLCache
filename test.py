from core.Request import RequestLoader
from envs.CacheEnv import CacheEnv

loader = RequestLoader(path="../RLCache/data/zipf", time_slot_length=10)


cache_env = CacheEnv(10, loader, 3)
cache_env.warm_up()
print(cache_env.cache.video_indices)
cache_env.step()