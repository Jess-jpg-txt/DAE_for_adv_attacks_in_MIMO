import numpy as np
np.random.seed(1337) # for reproducibility
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def fgsm_loss(model, data, epsilon):
    '''
    Perform the Fast Gradient Sign Method adversarial attack on the input data
    Parameters:
        model - a tensorflow model object
        data - original data without preturbation
        epsilon - preturbation magnitude
    Returns:
        data with preturbation generated with a FGSM attack
    '''
    data_t = tf.cast(data, tf.float32) # cast data to tensor
    with tf.GradientTape(persistent=True) as g:
        g.watch(data_t)
        loss = tf.reduce_sum(model(data_t)[:, 0:5], axis=0)
    grad = g.gradient(loss, data_t) # compute gradient of loss w.r.t input
    grad.numpy()
    grad_sign = np.sign(grad)
    # add adversarial perturbation and clip at cell edge
    x_adv = data_t + (epsilon * grad_sign)
    x_adv = np.clip(x_adv, a_min=np.min(data), a_max=(np.max(data)))
    return x_adv

def pgd(model, data, alpha, nb_iterations):
    '''
    Perform the Projected Gradient Descent adversarial attack on the input data
    Parameters:
        model - a tensorflow model object
        data - original data without preturbation
        alpha - preturbation magnitude
        nb_iterations - number of iterations to apply FGSM attack
    Returns:
        data with preturbation generated with a PGD attack
    '''
    x_adv = data # initialization
    for i in range(nb_iterations):
        x_adv = fgsm_loss(model=model, data=x_adv, epsilon=alpha)
    return x_adv


if __name__ == '__main__':
    '''The following code demonstrates how to use the above attack functions'''
    def create_relu_advanced(max_value=1.):
        def relu_advanced(x):
            return K.relu(x, max_value=K.cast_to_floatx(max_value))
        return relu_advanced

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def rel_mse(x_true, x_pred):
        loss = K.square(K.abs((x_true - x_pred)/ x_true))
        return K.mean(loss, axis=-1)

    precoder = 'MMMSE'
    model_num = '1'

    # Load input
    mat_contents = sio.loadmat('../data/testset_maxprod.mat')
    input = mat_contents['Input_tr_normalized']
    input = np.transpose(input)
    # load model
    reg_model = load_model(
        f'../saved_nn_models/model_{model_num}/ORIG_NN_{precoder}_maxprod_cell_1.h5', 
        custom_objects={'rel_mse':rel_mse, 'rmse':rmse})
    # make adversarial data
    x_adv_fgsm = fgsm_loss(model=reg_model, data=input, epsilon=0.2)
    print(x_adv_fgsm - input)
    x_adv_pgd = pgd(model=reg_model, data=input, alpha=0.01, nb_iterations=10)
    print(x_adv_pgd - input)