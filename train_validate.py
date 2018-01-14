import numpy as np
import pandas as pd
import os

def create_single_dataset(kind):
    df = pd.read_csv('preprocess/{0}/{0}.csv'.format(kind), index_col=0)
    samples = df.index.values

    X = None
    for s in samples:
        t = np.load('train/{}/tensor_aug3.npy'.format(str(s)))
        if X is None:
            X = t
        else:
            X = np.concatenate((X, t))
    np.save('preprocess/{}/tensor3'.format(kind), X)
    print(X.shape)
    if not kind == "test":
        y = df[['formation_energy_ev_natom', 'bandgap_energy_ev']].values
        y = np.repeat(y, 11, axis=0)
        np.save('preprocess/{}/target'.format(kind), y)

if __name__ == '__main__':
    create_single_dataset('train')
    create_single_dataset('validate')