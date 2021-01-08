"""
Q: Can we find a linear translation of Colored MNIST by minimizing an
adversarial loss between distributions, when the data is whitened?

A: Yes. This actually makes optimization a lot easier, because we can constrain
the map to be orthogonal.
"""

import procedures.translation

args = procedures.translation.make_args()
procedures.translation.main(args)