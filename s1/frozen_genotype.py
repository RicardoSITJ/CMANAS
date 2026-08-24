"""The frozen CMANAS-FER architecture (genotype) discovered by the paper's CMA-ES search.

Extracted from the baseline run's genotype.pickle (EXP-019). Kept as a Python literal so
revision experiments can rebuild the exact same architecture without re-running the search
(and without depending on a pickle path). Verified: init_channels=16, layers=8 -> 282,743 params.
"""

from genotypes import Genotype

# genotype.pickle from ricardorioda/cmanas-loso-2 -> outputs/.../eval-EXP-4000-100/genotype.pickle
CMANAS_FER = Genotype(
    normal=[
        ('sep_conv_3x3', 1), ('dil_conv_5x5', 0),
        ('sep_conv_5x5', 0), ('dil_conv_5x5', 1),
        ('dil_conv_3x3', 2), ('sep_conv_5x5', 3),
        ('max_pool_3x3', 4), ('max_pool_3x3', 1),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 1), ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1), ('dil_conv_3x3', 2),
        ('sep_conv_5x5', 1), ('skip_connect', 2),
        ('skip_connect', 0), ('dil_conv_5x5', 1),
    ],
    reduce_concat=range(2, 6),
)

# Architecture hyper-params used by the paper for this genotype (from the cmanas-loso notebook).
INIT_CHANNELS = 16
LAYERS = 8
