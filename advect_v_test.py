import numpy as np
import matplotlib.pyplot as plt

from mod4.tsai import tsai1d, tsai_FV
from mod4.implicit import advect_diffuse_IMPL as evolve
from mod4.utils import get_lin_mesh

from matplotlib.animation import FuncAnimation
from scipy.special import erf

i_pars = dict(Lv=8, dv=0.05, dt=1e-2, n_steps=10)
phy_pars = dict(omega_squared=1.0, gamma=1.1, sigma_squared=1.2**2)
x = 0
v = np.array(get_lin_mesh(i_pars))
p = np.exp(-((v))**2)

p = (np.abs(v)<1).astype(float)

print(p)
p /= np.trapz(p, v)

fig, ax = plt.subplots()
line, = ax.plot(v, p)

beta = phy_pars['gamma']/phy_pars['sigma_squared']
steady = np.exp(-beta*(v+phy_pars['omega_squared']*x/phy_pars['gamma'])**2)
steady /= np.trapz(steady, v)
plt.plot(v,steady, color="k", ls=":")
# plt.ylim(-5e-3,5e-3)

def update(i):
    global v, p
    p = evolve(p, x, phy_pars, i_pars)
    # p = np.array(p)
    # p /= np.sum(p)*i_pars['dv']
    print("norm",np.sum(p)*0.1)
    line.set_data(v, np.array(p))
    print("err", np.sqrt(np.mean((np.array(p)-steady)**2)))
    return line,


a = FuncAnimation(fig, update)




plt.show()