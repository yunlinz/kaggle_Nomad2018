import numpy as np
import pandas as pd
import os

def create_single_dataset(kind):
    df = pd.read_csv('preprocess/{0}/{0}.csv'.format(kind), index_col=0)
    samples = df.index.values
    n_samples = len(samples)
    X = None
    for i, s in enumerate(samples):
        t = np.load('train/{}/tensor_aug2.npy'.format(str(s)))
        if X is None:
            print(t.shape)
            _, d1, d2, d3, d4 = t.shape
            X = np.zeros((n_samples * 11, d1, d2, d3, d4), dtype=np.float32)
            print(X.shape)
        X[i*11:(i+1)*11,:,:,:,:] = t
    np.save('preprocess/{}/tensor2'.format(kind), X)
    y = df[['formation_energy_ev_natom', 'bandgap_energy_ev']].values
    y = np.repeat(y, 11, axis=0)
    np.save('preprocess/{}/target'.format(kind), y)

if __name__ == '__main__':
    create_single_dataset('train')
    create_single_dataset('validate')
    create_single_dataset('test')