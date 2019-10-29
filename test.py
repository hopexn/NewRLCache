from envs import *

cache_env = LfuCacheEnv(capacity=10, request_path="../RLCache/data/zipf", top_k=5, time_slot_length=100)
cache_env.reset()

print(cache_env.cache.get_content())

sum_hit_rate = 0
nb_steps = 1000
for i in range(nb_steps):
    hit_rate = cache_env.step()
    sum_hit_rate += hit_rate

print("hit rate: {}".format(sum_hit_rate / nb_steps))
