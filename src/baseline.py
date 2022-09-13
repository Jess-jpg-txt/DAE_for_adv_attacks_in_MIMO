# This code is adapted from 
# https://github.com/lucasanguinetti/Deep-Learning-Power-Allocation-in-Massive-MIMO/blob/master/python_code/NN_MMMSE_maxprod.py
# for [paper](https://arxiv.org/pdf/1812.03640.pdf)

import sys
import numpy as np
np.random.seed(1337) # for reproducibility
import scipy.io as sio
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda, Layer, Activation,Dropout,GaussianNoise
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import glorot_uniform
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model

def create_relu_advanced(max_value=1.):
    def relu_advanced(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_advanced

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def rel_mse(x_true, x_pred):
    loss = K.square(K.abs((x_true - x_pred)/ x_true))
    return K.mean(loss, axis=-1)

model_num = sys.argv[1]
precoder = sys.argv[2]
print(f"model {model_num}\nprecoder {precoder}")

cells = [1, 2, 3, 4]

for cell_index in cells:
    # Load input
    mat_contents = sio.loadmat('../data/dataset_maxprod.mat')
    print(mat_contents.keys())
    x_train = mat_contents['Input_tr_normalized']
    y_train = mat_contents[f'Output_tr_{precoder}_maxprod_cell_{cell_index}']

    x_train = np.transpose(x_train)
    y_train = np.transpose(y_train)
    print(f"x train shape {x_train.shape}")
    print(f"y train shape {y_train.shape}")
    n_sample = x_train.shape[0]
    n_feature = x_train.shape[1]

    # Maximum number of epochs
    N_max_epoch = 300
    N_batch_size = 256
    K_regularizer = None
    # Neural network configuration
    if model_num == "2":
        K_initializer = 'random_normal'
        B_initializer = 'random_uniform'
        model = Sequential()
        model.add(Dense(512, activation='elu', name = 'layer1', input_shape=(n_feature,), kernel_initializer=K_initializer, bias_initializer=B_initializer))
        model.add(Dense(256, activation='elu', name = 'layer2', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        model.add(Dense(128, activation='elu', name = 'layer3', kernel_initializer = K_initializer, bias_initializer=B_initializer))
        model.add(Dense(128, activation='elu', name = 'layer4', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        model.add(Dense(5, activation='elu', name = 'layer5', kernel_initializer = K_initializer, bias_initializer=B_initializer))
        model.add(Dense(6, activation='linear', name = 'layer6', trainable= False))
        model.get_layer('layer6').set_weights((np.column_stack([np.identity(5), np.ones(5)]), np.zeros(6)))
    elif model_num == "1":
        K_initializer = 'random_normal'
        B_initializer = 'random_uniform'
        model = Sequential()
        model.add(Dense(256, activation='elu', name = 'layer1', input_shape=(n_feature,), kernel_initializer=K_initializer, bias_initializer=B_initializer))
        model.add(Dense(128, activation='elu', name = 'layer2', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        model.add(Dense(64, activation='elu', name = 'layer3', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        model.add(Dense(64, activation='elu', name = 'layer4', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        model.add(Dense(5, activation='elu', name = 'layer5', kernel_initializer = K_initializer, bias_initializer=B_initializer))
        model.add(Dense(6, activation='linear', name = 'layer6', trainable= False))
        model.get_layer('layer6').set_weights((np.column_stack([np.identity(5), np.ones(5)]), np.zeros(6)))
    else:
        raise Exception("Model Number Not Accepted, Choose Either Model 1 or Model 2!")
    print(model.summary())

    # Optimizer
    adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1)

    # Early stopping and reduce learning rate
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0., patience=50, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    callback = [early_stopping, reduce_lr]
    model.compile(loss=rel_mse, optimizer='adam', metrics=[rmse])

    history = model.fit(x_train, y_train, validation_split=0.03125, epochs=N_max_epoch, batch_size=N_batch_size, callbacks=callback)
    model.save(f'../saved_nn_models/model_{model_num}/ORIG_NN_{precoder}_maxprod_cell_'+ str(cell_index) +'.h5')
