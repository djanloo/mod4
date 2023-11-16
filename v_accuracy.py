import numpy as np
import matplotlib.pyplot as plt
# from mod4.diffeq import advect_LW, diffuse_CN,  advect_diffuse_IMPL, advect_diffuse_LW, advectLW_diffuseCN
from mod4.tsai import tsai_FV as evolve
from mod4.implicit import advect_diffuse_IMPL
from mod4.utils import get_lin_mesh

import seaborn as sns; sns.set()
from matplotlib.animation import FuncAnimation
from scipy.special import erf

i_pars = dict(Lv=8, dv=0.1, dt=1e-2, n_steps = 100)
phy_pars = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)


v = np.array(get_lin_mesh(i_pars))
beta = phy_pars['gamma']/phy_pars['sigma_squared']

M = 20
x = 0.0
best_errs = np.zeros(M)

def err(x):
    return np.sqrt(np.mean((x**2)))



p = np.exp(-((v))**2)
p /= np.trapz(p, v)

steady = np.exp(-beta*(v+phy_pars['omega_squared']*x/phy_pars['gamma'])**2)
steady /= np.trapz(steady, v)

err_old = err(steady - p)
p = evolve(p, x, phy_pars, i_pars)
err_new = err(steady - p)

while abs(err_new - err_old)/err_old > 1e-7:
    p = evolve(p, x, phy_pars, i_pars)
    err_old = err_new
    err_new  = err(steady - p)
    print(f"{err_new:6.2e} -- {abs(err_new - err_old)/err_old:6.2e}")
print(f"tasi")
plt.plot(p-steady, label=f"I")

plt.legend()
plt.show()