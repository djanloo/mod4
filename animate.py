import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mod4.diffeq import funker_plank

# Settings
Lx, Lv = 4, 4
x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)

# initial conditions
x0, v0 = 1,  1
sx, sv = 0.2,  0.2
p = np.exp( -((X-x0)/sx)**2 - ((V-v0)/sv)**2)
p /= np.sum(p)*np.diff(x)[0]*np.diff(v)[0]

# integration & physical parameters
integration_params = dict(dt=np.pi/1000.0, n_steps=20)
physical_params = dict(alpha=1.0, gamma=0.2, sigma= 0.02, eps=0.1, omega=3, U0=0.1)

# What to plot
preproc_func = lambda x: x
levels = np.linspace(0,1.5, 30)

# Definition of the plots
fig, ax = plt.subplots()
cfplot = ax.contourf(X, V, preproc_func(p), levels=levels, cmap='rainbow')
cbar = fig.colorbar(cfplot, ax=ax)


def update(i):
    print(i, end ="-", flush=True)
    global cfplot, p
    # Sets the simulation to start at the last time
    physical_params['t0'] = i*integration_params['n_steps']*integration_params['dt']

    p , norm = funker_plank(p, x, v, physical_params, integration_params)

    p = np.array(p)
    p[p<0] = 0
    ax.clear()
    cfplot = ax.contourf(X, V, preproc_func(p), levels=levels, cmap='rainbow')
    return cfplot,


anim = FuncAnimation(fig, update, frames=300, interval=3/60*1e3, blit=False)
anim.save("anim6.mp4")
plt.show()
