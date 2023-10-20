from mod4.diffeq import funker_plank
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)

import matplotlib.pyplot as plt

Lx, Lv = 4, 4
x0, v0 = 0.0,  2.0
sx, sv = 0.6,  0.6

x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)
p0 = np.exp( -((X-x0)/sx)**2 - ((V-v0)/sv)**2)

p0 /= np.sum(p0)*np.diff(x)[0]*np.diff(v)[0]

for gamma in [2.1]:

    integration_args = dict(alpha=1.0, 
                            gamma=gamma, 
                            sigma= 0.8,
                    dt=np.pi/1000.0, n_steps=1500)

    period = np.pi*2*np.sqrt(1.0/integration_args['alpha'])
    print(f"period is roughly {period}")

    p , norm= funker_plank(p0, x, v, **integration_args)
    p = np.array(p)
    # p /= np.sum(p)*np.diff(x)[0]*np.diff(v)[0]

    p[p<0] = 0
    print(np.sum(np.isnan(p)))

    fig, (ax1, ax2)= plt.subplots(1,2, sharex=True, sharey=True, figsize=(8, 4))
    vmin = min(np.min(p0),np.min(p))
    vmax = max(np.max(p0), np.max(p))

    opt = dict(vmin=vmin, vmax=vmax)
    im = ax1.contourf(X, V, p0, **opt)
    ax1.set_title("t = 0")

    ax2.contourf(X,V, p, **opt)
    ax2.set_title(f"t = {integration_args['dt']*integration_args['n_steps']}")
    plt.colorbar(im, ax=ax2)

    plt.figure(2)
    plt.plot(np.abs(np.array(norm) - 1), label=f"gamma = {gamma:.2f}")
    plt.yscale('log')
    plt.ylabel(r"$|N_n - 1|$")
    plt.xlabel(r"n")
plt.legend()
plt.show()