import sys
import numpy as np
np.random.seed(1337) # for reproducibility
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Activation,Dropout,GaussianNoise
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from attacks import fgsm_loss, pgd

attack_type = sys.argv[2]
precoder = sys.argv[1]
model_num = sys.argv[3]
print(f"model: {model_num}\nprecoder: {precoder}\nattack type: {attack_type}")

def create_relu_advanced(max_value=1.):
    def relu_advanced(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_advanced
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def rel_mse(x_true, x_pred):
    loss = K.square(K.abs((x_true - x_pred)/ x_true))
    return K.mean(loss, axis=-1)

# Load input
mat_contents = sio.loadmat('../data/dataset_maxprod.mat')
input = mat_contents['Input_tr_normalized']
input = np.transpose(input)
print(f"input.shape: {input.shape}")

cells = [1, 2, 3, 4]
for cell_index in cells:
    model = load_model(
        f'../saved_nn_models/model_{model_num}/ORIG_NN_{precoder}_maxprod_cell_{cell_index}.h5', 
        custom_objects={'rel_mse':rel_mse, 'rmse':rmse})
    epsilon = 0.32
    # create adversarial attacks
    if attack_type == "PGD":
        nb_iterations = 10
        x_adv = pgd(model=model, data=input, alpha=epsilon/nb_iterations, nb_iterations=nb_iterations)
    elif attack_type == "FGSM":
        x_adv = fgsm_loss(model=model, data=input, epsilon=epsilon)

    # concatenate data
    total_output = np.concatenate([input, input])
    total_data = np.concatenate([input, x_adv])

    N_input = input.shape[1]  # Size of input vector
    N_max_epoch = 100  # Maximum number of epochs
    N_batch_size = 256  # Batch size
    K_initializer = 'random_normal'
    B_initializer = 'random_uniform'
    K_regularizer = None

    # Neural network configuration
    dae = Sequential()
    dae.add(Dense(40, activation='linear', name = 'input', input_shape=(N_input,), kernel_initializer=K_initializer, bias_initializer=B_initializer))
    dae.add(Dense(32, activation='linear', name = 'layer1', kernel_initializer =K_initializer, bias_initializer=B_initializer))
    dae.add(Dense(16, activation='linear', name = 'layer2', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    dae.add(Dense(32, activation='linear', name = 'layer3', kernel_initializer =K_initializer, bias_initializer=B_initializer))
    dae.add(Dense(N_input, activation='linear', name = 'layer4', trainable=False))

    # Optimizer
    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0., patience=50, verbose=0, mode='auto')
    callback = [early_stopping]
    
    # training
    dae.compile(loss='mean_squared_error', optimizer='adam')
    K.set_value(dae.optimizer.learning_rate, 0.001)
    history = dae.fit(total_data, total_output, validation_split=0.03125, epochs=N_max_epoch, batch_size=N_batch_size, callbacks=callback)

    dae.save(f'../saved_nn_models/model_{model_num}/DAE_{precoder}_{attack_type}_maxprod_cell_{cell_index}.h5')
