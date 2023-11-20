import numpy as np
import matplotlib.pyplot as plt

from mod4.tsai import  tsai_1D_x
from mod4.implicit import IMPL1D_x
from mod4.utils import get_lin_mesh

from matplotlib.animation import FuncAnimation
from scipy.special import erf


i_pars = dict(Lx=10, dx=0.1, dt=3e-3, n_steps=20)
phy_pars = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)
v = 1.0
x = np.array(get_lin_mesh(i_pars))
p = np.exp(-((x))**2)#*np.abs(np.sin(50*v))
# p = np.random.normal(0,1, size=len(p))
# p = (np.abs(v)<1).astype(float)

print(p)
p /= np.trapz(p, x)

fig, ax = plt.subplots()
line, = ax.plot(x, p)


P = np.zeros(len(p)-1)
for i in range(len(P)):
    P[i] = 0.5*(p[i] + p[i+1])

def update(i):
    global v, p, P
    p = IMPL1D_x(p, v, phy_pars, i_pars)

    p = np.array(p)
    p /= np.sum(p)*i_pars['dx']
    print("norm",np.sum(p)*0.1)
    line.set_data(x, np.array(p))
    return line,


a = FuncAnimation(fig, update)




plt.show()