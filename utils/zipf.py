import numpy as np
import pandas as pd

from numba import jit


def gen_data(path, dist_param=1.3, size=5000000):
    """
    生成zipf请求序列
    :param path: 生成结果保存路径
    :param dist_param:  分布参数
    :param size: 请求序列的数量
    """
    requests = pd.DataFrame()
    timestamps, video_ids = _gen_data(dist_param=dist_param, size=size)
    requests['timestamp'] = timestamps
    requests['video_id'] = video_ids
    requests.to_csv(path, index=False)


@jit(nopython=True, parallel=True)
def _gen_data(dist_param=1.3, size=5000000):
    video_ids = np.random.zipf(dist_param, size)
    timestamps = np.arange(1, size + 1)
    return timestamps, video_ids


def split_data(path, num_files):
    requests = pd.read_csv(path)
    num_requests = len(requests)
    pos = np.random.choice(range(num_files), size=num_requests, replace=True)
    for i in range(num_files):
        part_path = path + "_p{}".format(i)
        requests[pos == i].to_csv(part_path, index=False)