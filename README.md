# Defending Adversarial Attacks in MIMO Systems

This repository hosts code used to obtain results in our paper: [Defending Adversarial Attacks on Deep Learning Based Power Allocation in Massive MIMO Using Denoising Autoencoders]()

## Repository Structure
* `data/`: This folder contains the testing data set. Our work uses the publicly available [Power Allocation in Multi-Cell Massive MIMO dataset](https://data.ieeemlc.org/Ds2Detail). To download the training data set for our experiments, download the `multi_cell.zip` file, unzip and copy the file named `dataset_maxprod.mat` into the `data/` folder in this repository.
* `saved_nn_models/`: This folder contains saved neural network models from our experiments. Load these models to obtain the same results we showed in our paper. `model_1/` and `model_2/` sub-directories contain the saved models for model architecture 1 and model architecture 2 respectively, which are detailed in our paper.
* `src/` folder contains:
  * `attacks.py` that implements the adversarial attacks we use.
  * model architecture and training scripts: `baseline.py`, `dae_training.py` and `adv_regressor.py`.
  * result evaluation scripts: `eval_networks.py` to evaluate semi-whitebox experiments and `eval_blackbox.py` to evaluate blackbox experiments.
  * `Makefile` that facilitates running the experiments.
* `requirement.txt`: A snapshot of the Python package versions the experiments were run with.


## Get Started

### How to Use Saved Models to Reproduce Our Results

1. Ensure that you're in the `src/` folder:
```
$ pwd
DAE_for_adv_attacks_in_MIMO/src
```
2. Use the `Makefile` to run semi-whitebox experiments:
```
make eval_all
```
3. Use the `Makefile` to run blackbox experiments:
```
make eval_blackbox_all
```

### How to Re-train the Models

1. Ensure that you're in the `src/` folder:
```
$ pwd
DAE_for_adv_attacks_in_MIMO/src
```
2. Ensure that you have downloaded the dataset zip file from the [dataset website](https://data.ieeemlc.org/Ds2Detail), and have copied the training set into the `data/` folder as `/data/dataset_maxprod.mat`.
3. To re-train the baseline DL model:
```
make baseline_all
```
4. To re-train the DAE defense model:
```
make dae_all
```
5. To re-train the adversarially-trained regressor model:
```
make adv_regressor_all
```

### Package Requirements

To ensure success running of the program, the versions Python packages we used are listed in `requirements.txt`. To align the versions of your packages to this file, simply run:
```
pip install -r requirements.txt
```
