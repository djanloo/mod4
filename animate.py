import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mod4.diffeq import analytic
from mod4.diffeq import funker_plank

# Settings
Lx, Lv = 4, 4
x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)

# integration & physical parameters
integration_params = dict(dt=np.pi/100, n_steps=50)
physical_params = dict(omega_squared=1.0, gamma=2.1, sigma= 0.8)

# Initial conditions
x0, v0 = 0,  0
sx, sv = 0.2,  0.2
p = analytic(X,V, 0.1, x0, v0, physical_params)

# What to plot
preproc_func = lambda x: x
levels = np.linspace(0,1.5, 30)

# Definition of the plots
fig, ax = plt.subplots()
cfplot = ax.contourf(X, V, preproc_func(p), levels=levels, cmap='rainbow')
cbar = fig.colorbar(cfplot, ax=ax)

print("cazzo")
def update(i):
    print(i, end ="-", flush=True)
    global cfplot, p
    # Sets the simulation to start at the last time
    physical_params['t0'] = i*integration_params['n_steps']*integration_params['dt']

    p , norm = funker_plank(p, x, v, physical_params, integration_params)
    # p = analytic(X,V, (i + 30)*integration_params['dt'], x0, v0, physical_params)
    p = np.array(p)
    p[p<0] = 0
    ax.clear()
    cfplot = ax.contourf(X, V, preproc_func(p), levels=levels, cmap='rainbow')
    return cfplot,


anim = FuncAnimation(fig, update, frames=300, interval=3/60*1e3, blit=False)
# anim.save("anim_exact.mp4")
plt.show()
