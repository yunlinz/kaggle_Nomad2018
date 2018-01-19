from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os

def make_auxiliary_input():
    train_csv = 'preprocess/train/train.csv'
    validate_csv = 'preprocess/validate/validate.csv'
    test_csv = 'preprocess/test/test.csv'
    test_csv2 = 'test.csv'
    def auxiliary_input_helper(file, model=None):
        df = pd.read_csv(file)
        n = len(df)
        if model is None:
            model = OneHotEncoder()
            model.fit(df['spacegroup'].values.reshape((-1,1)))
        out_tensor = np.zeros((n, 16))

        out_tensor[:,:6] = model.transform(df['spacegroup'].values.reshape((-1,1))).todense()
        out_tensor[:,6] = df['number_of_total_atoms'].values / 80.0
        out_tensor[:,7:13] = df[['percent_atom_al',
                                 'percent_atom_ga',
                                 'percent_atom_in',
                                 'lattice_vector_1_ang',
                                 'lattice_vector_2_ang',
                                 'lattice_vector_3_ang']].values
        out_tensor[:,13:] = df[['lattice_angle_alpha_degree',
                                'lattice_angle_beta_degree',
                                'lattice_angle_gamma_degree']].values - 90

        np.save(file.split('.')[0] + '_aux', np.repeat(out_tensor, 11, axis=0))
        return model

    model = auxiliary_input_helper(train_csv, None)
    auxiliary_input_helper(validate_csv, model)
    auxiliary_input_helper(test_csv, model)
    auxiliary_input_helper(test_csv2, model)

if __name__ == '__main__':
    make_auxiliary_input()