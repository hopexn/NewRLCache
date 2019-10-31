import numba as nb

# 全局变量
NUM_FILTERS = 10


# 工具函数

@nb.jit(nopython=True, parallel=True)
def polyak_sum(weights, target_weights, polyak):
    return target_weights * polyak + weights * (1 - polyak)


def polyak_averaging(weights_list, target_weights_list, polyak):
    assert len(weights_list) == len(target_weights_list)
    nb_layers = len(weights_list)
    
    new_target_weights_list = []
    for idx in range(nb_layers):
        new_target_weights = polyak_sum(weights_list[idx], target_weights_list[idx], polyak)
        new_target_weights_list.append(new_target_weights)
    
    return new_target_weights_list
