import numpy as np
import matplotlib.pyplot as plt
from mod4.diffeq import advect_LW, diffuse_CN,  advect_diffuse_IMPL, advect_diffuse_LW, advectLW_diffuseCN
from mod4.utils import get_lin_mesh

import seaborn as sns; sns.set()
from matplotlib.animation import FuncAnimation
from scipy.special import erf

i_pars = dict(Lv=8, dv=0.1, dt=1e-3, n_steps = 500)
phy_pars = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)


v = np.array(get_lin_mesh(i_pars))
beta = phy_pars['gamma']/phy_pars['sigma_squared']

M = 20
x = 0.0
best_errs = np.zeros(M)

def err(x):
    return np.sqrt(np.mean((x**2)))


for diffCN in [True, False]:
    i_pars['diffCN']=diffCN

    p = np.exp(-((v))**2)
    p /= np.trapz(p, v)

    steady = np.exp(-beta*(v+phy_pars['omega_squared']*x/phy_pars['gamma'])**2)
    steady /= np.trapz(steady, v)

    err_old = err(steady - p)
    p = advect_diffuse_IMPL(p, x, phy_pars, i_pars)
    err_new = err(steady - p)

    while abs(err_new - err_old)/err_old > 1e-7:
        p = advect_diffuse_IMPL(p, x, phy_pars, i_pars)
        err_old = err_new
        err_new  = err(steady - p)
    print(f"IMPLICIT CN={diffCN}\terr={err_new}")
    plt.plot(steady - p, label=f"IMP CN={diffCN}")

for diffCN in [True, False]:
    i_pars['diffCN']=diffCN

    p = np.exp(-((v))**2)
    p /= np.trapz(p, v)

    steady = np.exp(-beta*(v+phy_pars['omega_squared']*x/phy_pars['gamma'])**2)
    steady /= np.trapz(steady, v)

    err_old = err(steady - p)
    p = advectLW_diffuseCN(p, x, phy_pars, i_pars)
    err_new = err(steady - p)
    norm = []
    while abs(err_new - err_old)/err_old > 1e-7:
        p = advect_LW(p, x, phy_pars, i_pars)
        p = advectLW_diffuseCN(p, x, phy_pars, i_pars)        
        err_old = err_new
        norm.append(np.trapz(p, v))
        err_new  = err(steady - p)
        print(err_new)
    plt.plot(np.log(norm))
    print(f"LW CN={diffCN}\terr={err_new}")
    plt.plot(steady - p, label=f"LW CN={diffCN}")
plt.legend()
plt.show()