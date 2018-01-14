import keras as K
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution3D, Flatten, MaxPooling3D, Dense, ZeroPadding3D
from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam, SGD, RMSprop, Nadam
import tensorflow as tf
import os

def create_graph(cell_size=17, load_file=None):
    model = Sequential()
    model.add(ZeroPadding3D((1,1,1), input_shape=(cell_size,cell_size,cell_size, 4)))
    model.add(Convolution3D(8, (3, 3, 3), activation='relu'))

    model.add(MaxPooling3D((2,2,2), strides=(2,2,2))) # shape should now be (8x8x8x8)

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Convolution3D(16, (3,3,3), activation='relu'))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2))) # shape should now be (16x4x4x4)

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Convolution3D(32, (3,3,3), activation='relu'))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2))) # shape should now be (32,2,2,2)

    model.add(Flatten()) # now should be a straight tensor of 256
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='relu'))

    if load_file is not None:
        model.load_weights(load_file)

    return model

def rmsle(actual, model):
    return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.log(model + 1) - tf.log(actual + 1)), axis=0)))

def train(continue_from=None, snapshot_freq=None, version="0.1", callbacks=None):
    optim = Nadam()
    if continue_from is not None:
        model = create_graph(load_file=continue_from)
    else:
        model = create_graph()
    model.compile(optimizer=optim, loss=rmsle)

    train_X = np.load('preprocess/train/tensor3.npy')
    train_y = np.load('preprocess/train/target.npy')

    validate_X = np.load('preprocess/validate/tensor3.npy')
    validate_y = np.load('preprocess/validate/target.npy')

    model.fit(train_X, train_y, validation_data=(validate_X, validate_y),
              shuffle=True, callbacks=callbacks, verbose=1, epochs=200, batch_size=64)

    if not os.path.exists('models/{}'.format(version)):
        os.mkdir('models')
        os.mkdir('models/{}'.format(version))

    model.save('models/{}/model_weights.h5'.format(version))

    return model

if __name__ == '__main__':
    filepath = 'models/0.2/model_weights_{epoch:03d}.h5'
    chechpoint = ModelCheckpoint(filepath, 'val_loss', verbose=1, save_best_only=True)
    model = train(callbacks=[chechpoint], version="0.2")

    validate_X = np.load('preprocess/validate/tensor3.npy')
    validate_y = np.load('preprocess/validate/target.npy')

    pred_y = model.predict(validate_X)
    import pandas as pd
    df = pd.DataFrame(validate_y, columns=['fe_actual', 'bg_actual'])
    df['fe_model'] = pred_y[:,0]
    df['bg_model'] = pred_y[:,1]

    df.to_csv('models/0.2/20_epoch_results.csv')

