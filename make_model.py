import keras as K
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Convolution3D, Flatten, MaxPooling3D, Dense, Input, concatenate, LeakyReLU, Dropout
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomUniform, TruncatedNormal

from keras.optimizers import Nadam
import tensorflow as tf
import os


def create_graph2(cell_size=27, load_file=None, l2_lambda=0.0, leakage=0.01, dropout=0.3):
    initializer = TruncatedNormal(0, 0.01)
    cell_input = Input((cell_size, cell_size, cell_size, 4), dtype='float32', name='crystall_cell')
    x = Convolution3D(6, (9,9,9), input_shape=(cell_size,cell_size,cell_size, 4),
                      activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda))(cell_input)
    x = LeakyReLU(leakage)(x)
    x = Convolution3D(6, (9,9,9), kernel_regularizer=K.regularizers.l2(l2_lambda), kernel_initializer=initializer)(x)
    x = Dropout(dropout)(x)
    x = LeakyReLU(leakage)(x)
    x = Convolution3D(6, (9,9,9), kernel_regularizer=K.regularizers.l2(l2_lambda), kernel_initializer=initializer)(x)
    x = Dropout(dropout)(x)
    x = LeakyReLU(leakage)(x)
    x = Flatten()(x)
    aux_input = Input((16,), name='aux_input')
    x = concatenate([x, aux_input])
    x = Dropout(dropout)(x)
    output = Dense(2, activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda), kernel_initializer=initializer)(x)
    model = Model(inputs=[cell_input, aux_input], outputs=[output])

    if load_file is not None:
        model.load_weights(load_file)

    optim = Nadam()
    model.compile(optimizer=optim, loss=rmsle)

    return model

def create_graph(cell_size=27, load_file=None, l2_lambda=0.01):
    model = Sequential()
    model.add(Convolution3D(8, (7,7,7), input_shape=(cell_size,cell_size,cell_size, 4), activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Convolution3D(16, (5,5,5), activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Convolution3D(32, (5,5,5), activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Convolution3D(64, (5,5,5), activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Convolution3D(128, (5,5,5), activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))
    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Dense(256, activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Dense(128, activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Dense(64, activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Dense(32, activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))
    model.add(Dense(2, activation='relu', kernel_regularizer=K.regularizers.l2(l2_lambda)))

    if load_file is not None:
        model.load_weights(load_file)
    optim = Nadam()
    model.compile(optimizer=optim, loss=rmsle)
    return model

def rmsle(actual, model):
    return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.log(model + 1) - tf.log(actual + 1)), axis=0)))

def train(continue_from=None, snapshot_freq=None, version="0.1", callbacks=None, epochs=10, l2_lambda=0):
    model = create_graph2(load_file=continue_from, l2_lambda=l2_lambda)


    train_X = np.concatenate((np.load('preprocess/train/tensor2.npy'), np.load('preprocess/validate/tensor2.npy')))
    aux_X = np.concatenate((np.load('preprocess/train/train_aux.npy'), np.load('preprocess/validate/validate_aux.npy')))

    train_y = np.concatenate((np.load('preprocess/train/target.npy'), np.load('preprocess/validate/target.npy')))

    validate_X = np.load('preprocess/test/tensor2.npy')
    aux_validate_X = np.load('preprocess/test/test_aux.npy')
    validate_y = np.load('preprocess/test/target.npy')

    model.fit([train_X, aux_X], train_y, validation_data=([validate_X, aux_validate_X], validate_y),
              shuffle=True, callbacks=callbacks, verbose=1, epochs=epochs, batch_size=64)

    if not os.path.exists('models/{}'.format(version)):
        os.mkdir('models')
        os.mkdir('models/{}'.format(version))

    model.save('models/{}/model_weights.h5'.format(version))

    return model

if __name__ == '__main__':
    version = "0.6"
    if not os.path.exists('models/' + version):
        os.mkdir('models/' + version)
    filepath = 'models/' + version + '/model_weights_{epoch:03d}.h5'
    chechpoint = ModelCheckpoint(filepath, 'val_loss', verbose=1, save_best_only=True)
    model = train(callbacks=[chechpoint], version=version, epochs=10)

    validate_X = np.load('preprocess/test/tensor2.npy')
    aux_X = np.load('preprocess/test/test_aux.npy')
    validate_y = np.load('preprocess/test/target.npy')

    pred_y = model.predict([validate_X, aux_X])
    import pandas as pd
    df = pd.DataFrame(validate_y, columns=['fe_actual', 'bg_actual'])
    df['fe_model'] = pred_y[:,0]
    df['bg_model'] = pred_y[:,1]

    df.to_csv('models/' + version +'/20_epoch_results.csv')

