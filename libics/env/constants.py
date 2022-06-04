from scipy import constants
from scipy.constants.constants import eV


###############################################################################


a_0 = constants.value("Bohr radius")
a0 = a_0
mu_B = constants.value("Bohr magneton")
muB = mu_B
mu_b = mu_B
mub = mu_B
eps_0 = constants.epsilon_0
eps0 = eps_0
epsilon0 = eps_0
k_B = constants.value("Boltzmann constant")
kB = k_B
k_b = k_B
kb = k_B


def __getattr__(name):
    return getattr(constants, name)
