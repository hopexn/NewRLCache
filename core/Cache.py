import numpy as np


class Cache:
    def __init__(self, capacity):
        self.capacity = capacity  # Cache容量
        
        # Cache中的内容，每一行表示一个元素的ID
        self.entries = np.zeros(shape=(capacity,), dtype=np.int) - 1
        
        self.indices = dict()  # 内容位置字典，用于查找元素的位置
    
    def clear(self):
        self.entries[:] = -1
        self.indices.clear()
    
    def length(self):
        return len(self.indices)
    
    def is_full(self):
        return self.length() >= self.capacity
    
    def find(self, element_id):
        """
        :param element_id: 目标元素的id
        :return: 如果元素在Cache中返回True，否则返回False
        """
        return element_id in self.indices
    
    def update(self, new_element, old_element=-1):
        if old_element == -1:
            assert self.length() < self.capacity
            element_idx = self.length()
            self.indices[new_element] = element_idx
            self.entries[element_idx] = new_element
        else:
            assert self.find(old_element)
            element_idx = self.indices[old_element]
            del self.indices[old_element]
            self.indices[new_element] = element_idx
            self.entries[element_idx] = new_element
    
    def get_content(self):
        return self.entries
