import numpy as np
import matplotlib.pyplot as plt
from mod4.diffeq import advect_v, diffuse_v,  advect_diffuse_IMPL, advect_LW as adlw, advectLW_diffuseCN, advect_diffuse_LW
from mod4.utils import get_lin_mesh

from matplotlib.animation import FuncAnimation
from scipy.special import erf

i_pars = dict(Lv=8, dv=0.1, dt=1e-3, n_steps=20, diffCN=True)
phy_pars = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8*0.8)
x = 0.0
v = np.array(get_lin_mesh(i_pars))
p = np.exp(-((v))**2)
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
    p = advectLW_diffuseCN(p, x, phy_pars, i_pars)
    line.set_data(v, np.array(p))
    print(np.sqrt(np.mean((np.array(p)-steady)**2)))
    return line,


a = FuncAnimation(fig, update)




plt.show()