import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pandas as pd
import pickle

from make_model import *


def make_submodel(outdir, test_data=None, epochs=25):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    train_csv = pd.read_csv('train.csv')
    sample = train_csv.sample(frac=1.0).copy()
    n_sample = len(sample)
    n_train = int(0.75 * n_sample)
    n_validate = n_sample - n_train
    sample['type'] = ['train'] * n_train + ['val'] * n_validate

    df_train = sample[sample['type'] == 'train']
    df_valid = sample[sample['type'] == 'val']

    inp, aux, train_target, ohe = make_input_output(df_train, None)
    valid_inp, valid_aux, valid_target, _   = make_input_output(df_valid, ohe)

    joblib.dump(ohe, outdir + '/ohe.pkl')

    model = create_graph2(27, None, 0.0, 0.0)
    model.fit([inp, aux], train_target,
              validation_data=([valid_inp, valid_aux], valid_target),
              batch_size=64, shuffle=True, epochs=epochs)

    model.save(outdir + '/model_weights.h5')

    def save_diagnostics(inpdata, auxdata, tardata, dfdata, name):
        y = model.predict([inpdata, auxdata])
        res = pd.DataFrame(y, columns=['fe_model', 'bg_model'])
        res['fe_actual'] = tardata[:, 0]
        res['bg_actual'] = tardata[:, 1]
        res['id'] = np.repeat(dfdata['id'].values, 11, axis=0)
        res.to_csv(name + '.csv', index=False)
        res.groupby('id').mean().to_csv(name + 'means.csv', index=False)

    save_diagnostics(inp, aux, train_target, df_train, outdir + '/train_diag')
    save_diagnostics(valid_inp, valid_aux, valid_target, df_valid, outdir + '/valid_diag')

    df_test = pd.read_csv('test.csv')
    inp, aux = None, None
    if test_data is not None:
        inp, aux = test_data
    else:
        inp, aux, _, _ = make_test_intput(df_test, ohe)

    y_test = model.predict([inp, aux])
    out_df = pd.DataFrame(df_test['id'], columns=['id'])
    out_df['formation_energy_ev_natom'] = np.mean(y_test[:, 0].reshape(-1, 11), axis=1)
    out_df['bandgap_energy_ev'] = np.mean(y_test[:, 1].reshape(-1, 11), axis=1)
    out_df.to_csv(outdir + '/submission.csv', index=False)


def make_test_intput(df, model):
    return make_input_output(df, model, True)


def make_input_output(df, model, isTest=False):
    if model is None:
        model = OneHotEncoder()
        model.fit(df['spacegroup'].values.reshape((-1,1)))
    ids = df['id'].values
    n_data = len(df)
    aux_tensor = np.zeros((n_data, 16))
    aux_tensor[:,:6] = model.transform(df['spacegroup'].values.reshape((-1,1))).todense()
    aux_tensor[:,6] = df['number_of_total_atoms'].values / 80.0
    aux_tensor[:,7:13] = df[['percent_atom_al',
                             'percent_atom_ga',
                             'percent_atom_in',
                             'lattice_vector_1_ang',
                             'lattice_vector_2_ang',
                             'lattice_vector_3_ang']].values
    aux_tensor[:,13:] = df[['lattice_angle_alpha_degree',
                            'lattice_angle_beta_degree',
                            'lattice_angle_gamma_degree']].values - 90
    aux_tensor = np.repeat(aux_tensor, 11, axis=0)

    inp_tensor = np.zeros((n_data * 11, 27, 27, 27, 4))
    for i, id in enumerate(ids):
        inp_tensor[i*11:(i+1)*11,:] = np.load('{}/{}/tensor_aug2.npy'.format('test' if isTest else 'train', id))

    if not isTest:
        target_tensor = np.repeat(df[['formation_energy_ev_natom', 'bandgap_energy_ev']].values, 11, axis=0)
    else:
        target_tensor = None

    return inp_tensor, aux_tensor, target_tensor, model



if __name__ == '__main__':
    name = "v0.9"
    submodels = 2

    outdir = 'output/' + name
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i in range(submodels):
        if not os.path.exists(outdir + '/{}'.format(str(i))):
            os.mkdir(outdir + '/{}'.format(str(i)))

    for i in range(1, submodels):
        print('submodel: {}'.format(i))
        make_submodel(outdir + '/' + str(i), epochs=15)
