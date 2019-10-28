import numpy as np


class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        
        # 写入计数
        self.write_count = 0
        
        # Cache中每一项分别保存视频ID
        self.entries = np.zeros(shape=(capacity,), dtype=np.int) - 1
        
        # 用于记录缓存的先后顺序
        self.write_counts = np.zeros(shape=(capacity,), dtype=np.int) - 1
        
        # 视频在缓存中的位置映射
        self.video_indices = dict()
    
    def __len__(self):
        return len(self.video_indices)
    
    def full(self):
        return len(self.video_indices) >= self.capacity
    
    def find(self, video_id):
        """
        在缓存中查找视频，若存在返回视频所在位置，若不存在则返回-1
        :param video_id: 待查找的视频的ID
        :return: 视频所在位置 或 -1（不存在）
        """
        if video_id not in self.video_indices:
            return -1
        else:
            return self.video_indices[video_id]
    
    def get(self, position):
        """
        访问缓存中指定位置的一个视频
        :param position:  访问视频的位置
        :return: 指定位置的视频
        """
        return self.entries[position]  # 返回访问视频的内容
    
    def put(self, position, video_id):
        """
        将视频存放在指定位置
        :param video_id:  待放置的视频
        :param position:  视频放置的位置
        :return: None
        """
        
        video_id_old = self.entries[position]
        if video_id_old in self.video_indices:
            del self.video_indices[video_id_old]
        
        self.video_indices[video_id] = position
        self.entries[position] = video_id  # 视频ID
        self.write_counts[position] = self.write_count
        
        self.write_count += 1
