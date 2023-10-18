from mod4.diffeq import funker_plank
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)

import matplotlib.pyplot as plt


x, v = np.linspace(-2 ,2, 80), np.linspace(-2, 2, 80)
dx, dv = np.diff(x)[0], np.diff(v)[0]

X, V = np.meshgrid(x,v)
p0 = np.exp( - (X/0.2)**2 - (V/0.2)**2)

p = funker_plank(p0, x, v, 
                alpha=0, gamma=0.1, sigma= 0.0,
                dt=0.01, n_steps=100)

plt.figure(1)
plt.contourf(X, V, p0)

plt.figure(2)
plt.contourf(X, V, p)
plt.show()