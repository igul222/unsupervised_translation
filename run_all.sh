#!/bin/bash

# Re-run every experiment in this project; they all save output to the outputs
# directory.

python 01_mnist_linear_translation.py
python 02_mnist_ica_marginals.py
python 03_mnist_ica_invariance.py
python 04_mnist_ica_seed_invariance.py
python 05_mnist_ica_permutation_recovery.py
python 06_mnist_translation_identity.py
python 07_mnist_translation_red_green.py
python 08_mnist_translation_whitened.py
python 09_tensor_decomp_toy.py