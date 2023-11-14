import contextlib
import itertools
import random
import time
import warnings
from typing import List

import numpy as np

with contextlib.suppress(Exception):
    from ELINA.python_interface.fconv import generate_sparse_cover

class GroupCache:
    groups = {}


def generate_sparse_cover0(subset_size: int, group_size: int, overlap_size: int) -> List[List[int]]:
    assert subset_size > group_size, "subset_size should ge greater than group_size."
    assert group_size > overlap_size, "overlap_size should be smaller than group_size."
    return generate_sparse_cover(subset_size, group_size, overlap_size)


def generate_sparse_cover00(subset_size: int, group_size: int, overlap_size: int) -> List[List[int]]:
    assert subset_size >= group_size, "subset_size should be not smaller than group_size."
    assert group_size > overlap_size, "overlap_size should be smaller than group_size."

    if (subset_size, group_size, overlap_size) in GroupCache.groups:
        return GroupCache.groups[(subset_size, group_size, overlap_size)]

    ids = list(range(subset_size))
    id_groups = []
    for new_group in itertools.combinations(ids, group_size):
        new_group = set(new_group)
        add = all(len(new_group.intersection(set(existed_group))) <= overlap_size for existed_group in id_groups)
        if add:
            id_groups.append(list(new_group))

    GroupCache.groups[(subset_size, group_size, overlap_size)] = id_groups
    return id_groups

def generate_sparse_cover000(subset_size: int, group_size: int, overlap_size: int) -> List[List[int]]:
    assert subset_size >= group_size, "subset_size should be not smaller than group_size."
    assert group_size > overlap_size, "overlap_size should be smaller than group_size."

    if (subset_size, group_size, overlap_size) in GroupCache.groups:
        return GroupCache.groups[(subset_size, group_size, overlap_size)]

    id_groups = []
    for start in range(0, subset_size, group_size - overlap_size):
        group = list(range(start, min(start + group_size, subset_size)))
        id_groups.append(group)

    GroupCache.groups[(subset_size, group_size, overlap_size)] = id_groups
    return id_groups


def generate_sparse_cover1(subset_size: int, group_size: int, overlap_size: int, multiple: int = 8) -> List[List[int]]:
    warnings.warn("This method is not ready.")
    assert subset_size >= group_size, "subset_size should be not smaller than group_size."
    assert group_size > overlap_size, "overlap_size should be smaller than group_size."

    if (subset_size, group_size, overlap_size, multiple) in GroupCache.groups:
        return GroupCache.groups[(subset_size, group_size, overlap_size, multiple)]

    groups_num = subset_size * multiple

    values_num = groups_num * group_size
    ub = subset_size

    values = np.abs(np.asarray(np.random.normal(0, 1, values_num) * 0.3 * ub, dtype=np.int32))
    values = np.hstack((values, np.arange(0, ub)))
    values[values >= ub] = ub - 1
    values.sort(axis=0)

    shuffled_values = values.copy()
    random.shuffle(shuffled_values)

    id_groups = []
    for i in range(groups_num):
        id_group = (shuffled_values[i * group_size:(i + 1) * group_size]).tolist()
        id_set = set(id_group)
        while len(set(id_set)) != group_size:
            id_set.add(random.randint(0, subset_size))
        id_groups.append(id_group)

    # for i in range(groups_num):
    #     for j in range(i + 1, groups_num):
    #         intersection = set(id_groups[i]).intersection(set(id_groups[j]))
    #         if len(intersection) > group_size - 2:
    #             print(intersection)
    # frequences = [0] * subset_size
    # for g in id_groups:
    #     for a in g:
    #         frequences[a] += 1
    #
    # plt.plot(frequences)
    # plt.show()
    GroupCache.groups[(subset_size, group_size, overlap_size, multiple)] = id_groups

    return id_groups


MAX_BASIC_GROUP_SIZE = 4
def generate_sparse_cover2(subset_size: int, group_size: int, overlap_size: int) -> List[List[int]]:
    warnings.warn("This method is not ready.")
    assert subset_size >= group_size, "subset_size should be not smaller than group_size."
    assert group_size > overlap_size, "overlap_size should be smaller than group_size."

    ids = list(range(subset_size))
    id_groups = []

    basic_group_size = min(group_size, MAX_BASIC_GROUP_SIZE)
    for new_group in itertools.combinations(ids, basic_group_size):
        new_group = set(new_group)
        add = all(len(new_group.intersection(set(existed_group))) <= overlap_size for existed_group in id_groups)

        if add:
            id_groups.append(list(new_group))

    if group_size > MAX_BASIC_GROUP_SIZE:
        assert subset_size > group_size, "subset_size should be greater than group_size."
        id_groups2 = []
        for group in id_groups:
            while True:
                new_group = set(group)
                while len(new_group) < group_size:
                    new_group.add(random.randint(0, subset_size - 1))
                add = all(len(new_group.intersection(set(existed_group))) <= group_size - 1 for existed_group in id_groups2)

                if add:
                    id_groups2.append(list(new_group))
                    break
        id_groups = id_groups2

    return id_groups

if __name__ == '__main__':
    # start = time.time()
    # groups = generate_sparse_cover00(100, 3, 1)
    # print(time.time() - start)
    # print(groups)
    GroupCache.groups = {}
    start = time.time()
    groups = generate_sparse_cover000(100, 3, 1)
    print(time.time() - start)
    print(groups)