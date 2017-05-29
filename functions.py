# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(vector, gain=5):
    return 1/(1+np.exp(-gain*vector))


def error(out, label):
    # logエラーよけ
    out[out == 0] = 0.0000001
    out[out == 1] = 0.9999999
    # 誤差関数によりエラーをだす
    err = label*np.log(out) + (1-label)*np.log(1-out)
    err[np.isnan(label)] = 0
    # 有効な値の個数
    n = len(label)-sum(np.isnan(label))
    if 0 == n:
        return np.nan
    return -np.sum(err)/n

