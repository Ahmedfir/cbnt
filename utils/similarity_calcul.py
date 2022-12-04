from enum import Enum
from typing import List


class ShorteningStrategy(Enum):
    FIRST_ITEM = 0  # not implemented
    LAST_ITEM = 1


class SizeFitter:

    def __init__(self, items_arr, size, filling_item, shortening_strategy=ShorteningStrategy.LAST_ITEM):
        self.items_arr = items_arr
        self.size = size
        self.filling_item = filling_item
        self.shortening_strategy = shortening_strategy

    def _largest_items_size_diff(self):
        return max([len(l) - self.size for l in self.items_arr])

    def _fill_to_size(self, arr: List):
        while len(arr) < self.size:
            arr.append(self.filling_item)
        return arr

    def fit(self):
        # remove same number of tokens from the end of both sequences:
        # get biggest diff
        max_diff = self._largest_items_size_diff()
        while max_diff > 0:
            self.items_arr = list(map(lambda arr: arr[:-max_diff], self.items_arr))
            max_diff = self._largest_items_size_diff()
        self.items_arr = list(map(lambda arr: self._fill_to_size(arr), self.items_arr))
        return self.items_arr


def cosine_similarity_chunk(cosine_similarity_func, mat, list_mats):
    il = len(list_mats)
    assert il > 0
    res = [cosine_similarity_func(mat, m) for m in list_mats]
    assert il == len(res)
    return res


def torch_cosine(matrix1, matrix2):
    import torch.nn.functional as F
    return F.cosine_similarity(matrix1.flatten(), matrix2.flatten(), dim=0).tolist()


def scipy_cosine(matrix1, matrix2):
    # delta_time = DeltaTime()
    import numpy as np
    from scipy.spatial import distance
    arr1 = np.concatenate(matrix1.detach().numpy(), axis=0)
    # delta_time.print('cosine_similarity_mats arr1')
    arr2 = np.concatenate(matrix2.detach().numpy(), axis=0)
    # delta_time.print('cosine_similarity_mats arr2')
    dis = distance.cosine(arr1, arr2)
    sim = 1 - dis
    # delta_time.print('cosine_similarity_mats sim')
    return sim

