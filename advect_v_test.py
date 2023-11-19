import numpy as np
import matplotlib.pyplot as plt

from mod4.tsai import  tsai_FV, tsai_I, tsai_2
from mod4.implicit import advect_diffuse_IMPL
from mod4.utils import get_lin_mesh

from matplotlib.animation import FuncAnimation
from scipy.special import erf

i_pars = dict(Lv=10, dv=0.1, dt=3e-3, n_steps=20)
phy_pars = dict(omega_squared=1.0, gamma=1.1, sigma_squared=0.8**2)
x = 0.5
v = np.array(get_lin_mesh(i_pars))
p = np.exp(-((v))**2)#*np.abs(np.sin(50*v))
# p = np.random.normal(0,1, size=len(p))
# p = (np.abs(v)<1).astype(float)

print(p)
p /= np.trapz(p, v)

fig, ax = plt.subplots()
line, = ax.plot(v, p)

beta = phy_pars['gamma']/phy_pars['sigma_squared']
steady = np.exp(-beta*(v+phy_pars['omega_squared']*x/phy_pars['gamma'])**2)
steady /= np.trapz(steady, v)
plt.plot(v,steady, color="k", ls=":")
# plt.ylim(-5e-3,5e-3)

evolve = tsai_FV
# evolve = advect_diffuse_IMPL

def update(i):
    global v, p
    p = evolve(p, x, phy_pars, i_pars)
    p = np.array(p)
    p /= np.sum(p)*i_pars['dv']
    print("norm",np.sum(p)*0.1)
    line.set_data(v, np.array(p))
    print("err", np.sqrt(np.mean((np.array(p)-steady)**2)))
    return line,


a = FuncAnimation(fig, update)




plt.show()