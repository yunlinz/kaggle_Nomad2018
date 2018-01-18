import make_model
import os
import pandas as pd
import numpy as np

def model_test(file, version):
    model = make_model.create_graph(load_file=file)
    model.compile(loss=make_model.rmsle, optimizer='nadam')
    results = []
    for i in os.listdir('test'):
        X = np.load('test/{}/tensor_aug2.npy'.format(i))
        y = model.predict(X).mean(axis=0)
        results.append([int(i), y[0], y[1]])
    df = pd.DataFrame(results, columns=['id','formation_energy_ev_natom','bandgap_energy_ev'])
    df.sort_values('id').to_csv('submission_{}.csv'.format(version), index=False)

if __name__ == '__main__':
    model_test('models/0.4c/model_weights.h5', "0.4d")