import random

import numpy as np


def decision(probability):
    return random.random() < probability


def distance_to_line(p, p1, p2):
    return np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)


def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))  # edit
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans)


def lin2db(linear_input):
    return 10 * np.log10(linear_input)


def db2lin(db_input):
    return 10 ** (db_input / 10)


def numpy_object_array(object, dtype=None):
    """1D for now only"""
    obj_len = len(object)
    object_type = type(object[0]) if obj_len > 1 else type(object)
    arr = np.empty(obj_len, dtype=object_type)
    if obj_len < 2:
        arr[0] = object
        return arr
    else:
        for i in range(obj_len):
            arr[i] = object[i]
        return arr
