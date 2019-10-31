import matplotlib.pyplot as plt
import numpy as np

from envs import *
from utils.utils import *

init_tf()

cache_env = SacCacheEnv(capacity=10, request_path="./data/zipf", top_k=3, time_slot_length=50)
cache_env.reset()

print(cache_env.cache.get_content())

hit_rate_list = []
nb_steps = 100000
for i in range(nb_steps):
    hit_rate = cache_env.step()
    hit_rate_list.append(hit_rate)
    print("Step {:5d}: {:.2f}, {:.2f}".format(i, hit_rate, np.mean(hit_rate_list)))

print("hit rate: {:.2f}".format(np.mean(hit_rate_list[-1000:])))

plt.plot(hit_rate_list)
plt.xlabel("step")
plt.ylabel("hit rate")
plt.figure()
plt.pause(10)
