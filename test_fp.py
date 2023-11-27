from mod4.diffeq import funker_plank
from mod4.utils import quad_int
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
from time import perf_counter

import matplotlib.pyplot as plt

## Environment settings
Lx, Lv = 4, 4

x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)

# initial condition
x0, v0 = 0, 0
sx, sv = 0.6,  0.6
p0 = np.exp( -((X-x0)/sx)**2 - ((V-v0)/sv)**2)
p0 /= quad_int(p0, x, v)

# Integration settings & physical parameters
integration_params = dict(dt=np.pi/1000.0, n_steps=1000)
physical_params = dict(omega_squared=1.0, gamma=2.1, sigma= 0.8)

## Figure
fig, axes = plt.subplot_mosaic([["init", "cbar", "evol"]], width_ratios=[1,0.1,1], constrained_layout=True)



p , norm, curr = funker_plank(p0, x, v, physical_params, integration_params, save_norm=True)

p = np.array(p)


vmin = min(np.min(p0),np.min(p))
vmax = max(np.max(p0), np.max(p))
opt = dict(vmin=vmin, vmax=vmax)

im = axes['init'].contourf(X, V, p0, **opt)
im = axes['evol'].contourf(X, V, p, **opt)

plt.colorbar(im, cax=axes['cbar'])



axes['init'].set_title("t = 0")
axes['evol'].set_title(f"t = {integration_params['dt']*integration_params['n_steps']}")

plt.figure(2)
plt.plot(np.abs(np.array(norm) - 1))
plt.yscale('log')
plt.ylabel(r"$|N_n - 1|$")
plt.xlabel(r"n")

print(f"difference is {np.sum(np.abs(p0 - p))/80/80}")

plt.show()