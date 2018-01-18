import os
from util import *
import numpy as np
import multiprocessing

def augment_date(filename, repeats=10, fineness=1):
    sc = super_cell(filename)
    t0 = np.asarray([sc.to_tensor(fineness=fineness)])
    for _ in range(repeats):
        t0 = np.concatenate((t0,
                             np.asarray([
                                 sc.random_transform().to_tensor(fineness=fineness,
                                                                 gamma_al=1.0,
                                                                 gamma_ga=1.0,
                                                                 gamma_in=1.0,
                                                                 gamma_o=1.0)])))
    return t0


def augment_data2(filename, repeats=10, fineness=1):
    sc = super_cell(filename)
    t0 = np.asarray([sc.to_tensor2(fineness=fineness, cell_size=13)])
    for _ in range(repeats):
        t0 = np.concatenate((t0,
                             np.asarray([
                                 sc.random_transform().to_tensor2(fineness=fineness, cell_size=13)])))
    return t0

def mk_aug_data(f):
    tensor = augment_data2('test/{}/geometry.xyz'.format(f))
    np.save('test/{}/tensor_aug2'.format(f), tensor)

if __name__ == '__main__':
    pool = multiprocessing.Pool(7)
    pool.map(mk_aug_data, os.listdir('test'))