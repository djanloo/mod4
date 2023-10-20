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
sx, sv = 0.4,  0.4
p = np.exp( -((X-x0)/sx)**2 - ((V-v0)/sv)**2)
p /= np.sum(p)*np.diff(x)[0]*np.diff(v)[0]

# integration & physical parameters
integration_args = dict(    alpha=1.0, 
                            gamma=2.1, 
                            sigma= 0.2,
                            dt=np.pi/1000.0, n_steps=20)
# What to plot
preproc_func = lambda x: x
levels = np.linspace(0,1, 30)

# Definition of the plots
fig, ax = plt.subplots()
cfplot = ax.contourf(X, V, preproc_func(p), levels=levels)
cbar = fig.colorbar(cfplot, ax=ax)

def update(i):
    global cfplot, p
    p , norm = funker_plank(p, x, v, **integration_args)

    p = np.array(p)
    p[p<0] = 0
    ax.clear()
    cfplot = ax.contourf(X, V, preproc_func(p), levels=levels)
    return cfplot,


anim = FuncAnimation(fig, update, frames=200, interval=3/60*1e3, blit=False)
anim.save("anim3.mp4")
plt.show()
