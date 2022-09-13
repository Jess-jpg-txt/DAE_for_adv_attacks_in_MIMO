import sys
import numpy as np
np.random.seed(1337) # for reproducibility
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Activation,Dropout,GaussianNoise
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers
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

# load trained classifier
cells = [1, 2, 3, 4]

for cell_index in cells:

    y_train = mat_contents[f'Output_tr_{precoder}_maxprod_cell_' + str(cell_index)]
    y_train = np.transpose(y_train)

    model = load_model(
        f'../saved_nn_models/model_{model_num}/ORIG_NN_{precoder}_maxprod_cell_{cell_index}.h5', 
        custom_objects={'rel_mse':rel_mse, 'rmse':rmse})
    # create adversarial attacks
    epsilon = 0.025
    if attack_type == "PGD":
        nb_iterations = 10
        x_adv = pgd(model=model, data=input, alpha=epsilon/nb_iterations, nb_iterations=nb_iterations)
    elif attack_type == "FGSM":
        x_adv = fgsm_loss(model=model, data=input, epsilon=epsilon)

    # concatenate data
    total_y = np.concatenate([y_train, y_train])
    total_x = np.concatenate([input, x_adv])

    n_feature = total_x.shape[1]
    N_output = total_y.shape[1]
    N_max_epoch = 50  # Maximum number of epochs
    N_batch_size = 256  # Batch size
    K_initializer = 'random_normal'
    B_initializer = 'random_uniform'
    K_regularizer = None

    # Neural network configuration
    if model_num == "2":
        adv_regressor = Sequential()
        adv_regressor.add(Dense(512, activation='elu', name = 'layer1', input_shape=(n_feature,), kernel_initializer=K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(256, activation='elu', name = 'layer2', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(128, activation='elu', name = 'layer3', kernel_initializer = K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(128, activation='elu', name = 'layer4', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(5, activation='elu', name = 'layer5', kernel_initializer = K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(6, activation='linear', name = 'layer6', trainable= False))
        adv_regressor.get_layer('layer6').set_weights((np.column_stack([np.identity(5), np.ones(5)]), np.zeros(6)))
    elif model_num == "1":
        adv_regressor = Sequential()
        adv_regressor.add(Dense(256, activation='elu', name = 'layer1', input_shape=(n_feature,), kernel_initializer=K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(128, activation='elu', name = 'layer2', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(64, activation='elu', name = 'layer3', kernel_initializer = K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(64, activation='elu', name = 'layer4', kernel_initializer =K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(5, activation='elu', name = 'layer5', kernel_initializer = K_initializer, bias_initializer=B_initializer))
        adv_regressor.add(Dense(6, activation='linear', name = 'layer6', trainable= False))
        adv_regressor.get_layer('layer6').set_weights((np.column_stack([np.identity(5), np.ones(5)]), np.zeros(6)))
    else:
        raise Exception("Model Number Not Accepted, Choose Either Model 1 or Model 2!")
    print(adv_regressor.summary())

    # Optimizer
    adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1)

    # Early stopping and reduce learning rate
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0., patience=50, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    callback = [early_stopping, reduce_lr]
    adv_regressor.compile(loss=rel_mse, optimizer='adam', metrics=[rmse])
    
    # train and save model
    history = adv_regressor.fit(total_x, total_y, validation_split=0.03125, epochs=N_max_epoch, batch_size=N_batch_size, callbacks=callback)
    adv_regressor.save(f'../saved_nn_models/model_{model_num}/ADV_REGRESSOR_0025_{precoder}_{attack_type}_maxprod_cell_{cell_index}.h5')
