"""
Is ICA approximately invariant to noise and orthogonal transformations, up to
permutations / flips? Yes!
"""

import torch
from lib import utils, ops, datasets

# TODO re-implement this experiment; it was lost in a refactor.