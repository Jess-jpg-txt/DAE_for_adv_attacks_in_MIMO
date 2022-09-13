# Python
PYTHON = python3
# experiment setting variables
PRECODER = MR
ATTACK = PGD
MODELNUM = 1

#############################
# train baseline models
baseline_all:
	$(PYTHON) baseline.py 1 MMMSE
	$(PYTHON) baseline.py 2 MMMSE
	$(PYTHON) baseline.py 1 MR
	$(PYTHON) baseline.py 2 MR

#############################
# train DAE models
dae_all:
	$(PYTHON) dae_training.py MMMSE FGSM 1
	$(PYTHON) dae_training.py MMMSE PGD 1
	$(PYTHON) dae_training.py MR FGSM 1
	$(PYTHON) dae_training.py MR PGD 1
	$(PYTHON) dae_training.py MMMSE FGSM 2
	$(PYTHON) dae_training.py MMMSE PGD 2
	$(PYTHON) dae_training.py MR FGSM 2
	$(PYTHON) dae_training.py MR PGD 2

#############################
# train adversarially-trained regressors
adv_regressor_all:
	$(PYTHON) adv_regressor.py MMMSE FGSM 1
	$(PYTHON) adv_regressor.py MMMSE PGD 1
	$(PYTHON) adv_regressor.py MR FGSM 1
	$(PYTHON) adv_regressor.py MR PGD 1
	$(PYTHON) adv_regressor.py MMMSE FGSM 2
	$(PYTHON) adv_regressor.py MMMSE PGD 2
	$(PYTHON) adv_regressor.py MR FGSM 2
	$(PYTHON) adv_regressor.py MR PGD 2

#############################
# run all whitebox experiments
eval_all:
	$(PYTHON) eval_networks.py MMMSE FGSM 1
	$(PYTHON) eval_networks.py MMMSE PGD 1
	$(PYTHON) eval_networks.py MR FGSM 1
	$(PYTHON) eval_networks.py MR PGD 1
	$(PYTHON) eval_networks.py MMMSE FGSM 2
	$(PYTHON) eval_networks.py MMMSE PGD 2
	$(PYTHON) eval_networks.py MR FGSM 2
	$(PYTHON) eval_networks.py MR PGD 2

#############################
# run all the blackbox experiments
eval_blackbox_all:
	$(PYTHON) eval_blackbox.py MMMSE FGSM 1
	$(PYTHON) eval_blackbox.py MMMSE PGD 1
	$(PYTHON) eval_blackbox.py MR FGSM 1
	$(PYTHON) eval_blackbox.py MR PGD 1
	$(PYTHON) eval_blackbox.py MMMSE FGSM 2
	$(PYTHON) eval_blackbox.py MMMSE PGD 2
	$(PYTHON) eval_blackbox.py MR FGSM 2
	$(PYTHON) eval_blackbox.py MR PGD 2
