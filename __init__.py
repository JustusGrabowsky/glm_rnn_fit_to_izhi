"""
GLM Python Port - Faithful translation of MATLAB GLM_and_Izhikevich code

This package contains a Python implementation of the GLM fitting code from
Weber & Pillow 2017 (Neural Computation), "Capturing the dynamical repertoire
of single neurons with generalized linear models".

The code has been translated as faithfully as possible from the original MATLAB
implementation found in GLM_and_Izhikevich/.
"""

from .sameconv import sameconv
from .normalizecols import normalizecols
from .logexp1 import logexp1
from .makeBasis_StimKernel import makeBasis_StimKernel
from .makeBasis_PostSpike import makeBasis_PostSpike
from .negloglike_glm_basis import negloglike_glm_basis
from .negloglike_glm_basis_softRect import negloglike_glm_basis_softRect
from .fit_glm import fit_glm
from .simulate_glm import simulate_glm
from .simulate_izhikevich import simulate_izhikevich
from .generate_izhikevich_stim import generate_izhikevich_stim

__all__ = [
    'sameconv',
    'normalizecols',
    'logexp1',
    'makeBasis_StimKernel',
    'makeBasis_PostSpike',
    'negloglike_glm_basis',
    'negloglike_glm_basis_softRect',
    'fit_glm',
    'simulate_glm',
    'simulate_izhikevich',
    'generate_izhikevich_stim',
]
