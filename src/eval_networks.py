import sys
import os
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from attacks import fgsm_loss, pgd

attack_type = sys.argv[2]
precoder = sys.argv[1]
model_num = sys.argv[3]
print(f"model: {model_num}\nprecoder: {precoder}\nattack type: {attack_type}")

# load data
mat_contents = sio.loadmat('../data/testset_maxprod.mat')
x_test = mat_contents['Input_tr_normalized']
x_test = np.transpose(x_test)
print(f"x_test.shape: {x_test.shape}")  # (5000, 40)
print(np.mean(x_test))
print(x_test)

# load model
def create_relu_advanced(max_value=1.):
    def relu_advanced(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_advanced
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def rel_mse(x_true, x_pred):
    loss = K.square(K.abs((x_true - x_pred)/ x_true))
    return K.mean(loss, axis=-1)

cells = [1,2,3,4]
n_cells = len(cells)
# invalid prediction percentage lists init
adv = []
adv_dae = []
adv_regress = []
dim_reduce = []
epsilons = np.linspace(0.01, 1, 30)

for epsilon in epsilons:
    invalid_perc_base = 0
    invalid_perc_adv = 0
    invalid_perc_adv_dae = 0
    invalid_perc_adv_regressor = 0
    invalid_perc_dim_reduce = 0
    for cell_index in cells:
        model = load_model(
            f'../saved_nn_models/model_{model_num}/ORIG_NN_{precoder}_maxprod_cell_{cell_index}.h5', 
            custom_objects={'rel_mse':rel_mse, 'rmse':rmse})
        # regardless of what the attack is, always use the DAE model trained with PGD attck 
        model_dae = load_model(
            f'../saved_nn_models/model_{model_num}/DAE_{precoder}_PGD_maxprod_cell_{cell_index}.h5')
        model_adv_regressor = load_model(
            f'../saved_nn_models/model_{model_num}/ADV_REGRESSOR_0025_{precoder}_{attack_type}_maxprod_cell_{cell_index}.h5',
            custom_objects={'rel_mse':rel_mse, 'rmse':rmse})

        ### evaluate case 1: original model + original input 
        if (epsilon == epsilons[0]):
            output_NN = model.predict(x_test)
            total_invalid = np.sum(output_NN[:, 0:5], axis=1) > 500
            invalid_perc_base += np.sum(total_invalid)

        ## apply the adversarial attacks
        if attack_type == "PGD":
            nb_iterations = 10
            # attack for original DL model and DAE defense
            x_adv = pgd(model=model, data=x_test, alpha=epsilon/nb_iterations, nb_iterations=nb_iterations)
            # attack for the adversarially trained model
            x_adv_adv_regres = pgd(model=model_adv_regressor, data=x_test, alpha=epsilon/nb_iterations, nb_iterations=nb_iterations)
        elif attack_type == "FGSM":
            # attack for original DL model and DAE defense
            x_adv = fgsm_loss(model=model, data=x_test, epsilon=epsilon)
            # attack for the adversarially trained model
            x_adv_adv_regres = fgsm_loss(model=model_adv_regressor, data=x_test, epsilon=epsilon)

        ### evaluate case 2: original model + adversarial input
        output_NN_adv = model.predict(x_adv)
        total_invalid_adv = np.sum(output_NN_adv[:, 0:5], axis=1) > 500
        invalid_perc_adv += np.sum(total_invalid_adv)

        ### evaluate case 3: Denoising Autoencoder model + adversarial input
        # first get denoised output from DAE
        input_denoised = model_dae.predict(x_adv)
        # then feed the output of DAE to original regressor model
        output_NN_adv_dae = model.predict(input_denoised)   # (5000, 6)
        total_invalid_adv_dae = np.sum(output_NN_adv_dae[:, 0:5], axis=1) > 500
        invalid_perc_adv_dae += np.sum(total_invalid_adv_dae)

        ### evaluate case 4: Adversarial regressor model + adversarial input
        output_NN_adv_regressor = model_adv_regressor.predict(x_adv_adv_regres)
        total_invalid_adv_regressor = np.sum(output_NN_adv_regressor[:, 0:5], axis=1) > 500
        invalid_perc_adv_regressor += np.sum(total_invalid_adv_regressor)

    print('-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print(f'@ epsilon={epsilon:.2f}, Percentage invalid output:')
    
    # case 1:
    if (epsilon == epsilons[0]):
        invalid_perc_base = (invalid_perc_base * 100) / (len(x_test) * n_cells)
        print(f"Original Model + Original Input: {invalid_perc_base: .2f}%")
        base = [invalid_perc_base] * len(epsilons)
    # case 2:
    invalid_perc_adv = (invalid_perc_adv * 100) / (len(x_adv) * n_cells)
    print(f"Original Model + Adversarial Input: {invalid_perc_adv: .2f}%")
    adv.append(invalid_perc_adv)
    # case 3:
    invalid_perc_adv_dae = (invalid_perc_adv_dae * 100) / (len(x_adv) * n_cells)
    print(f"DAE Model + Adversarial Input: {invalid_perc_adv_dae: .2f}%")
    adv_dae.append(invalid_perc_adv_dae)
    # case 4:
    invalid_perc_adv_regressor = (invalid_perc_adv_regressor * 100) / (len(x_adv) * n_cells)
    print(f"Adv Regressor + Adversarial Input: {invalid_perc_adv_regressor: .2f}%")
    adv_regress.append(invalid_perc_adv_regressor)

if not os.path.isdir('../eval_output'):
    os.makedirs('../eval_output')
if not os.path.isdir(f'../eval_output/white_box_model_{model_num}'):
    os.makedirs(f'../eval_output/white_box_model_{model_num}')
if not os.path.isdir(f'../eval_output/white_box_model_{model_num}/{precoder}_{attack_type}'):
    os.makedirs(f'../eval_output/white_box_model_{model_num}/{precoder}_{attack_type}')
np.savetxt(f"../eval_output/white_box_model_{model_num}/{precoder}_{attack_type}/epsilons.txt", epsilons)
np.savetxt(f"../eval_output/white_box_model_{model_num}/{precoder}_{attack_type}/base.txt", base)
np.savetxt(f"../eval_output/white_box_model_{model_num}/{precoder}_{attack_type}/adv.txt", adv)
np.savetxt(f"../eval_output/white_box_model_{model_num}/{precoder}_{attack_type}/adv_dae.txt", adv_dae)
np.savetxt(f"../eval_output/white_box_model_{model_num}/{precoder}_{attack_type}/adv_regressor.txt", adv_regress)
