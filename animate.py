import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mod4.utils import analytic
from mod4.implicit import  generic_3_step
from mod4.utils import quad_int, get_quad_mesh, get_lin_mesh
from mod4.tsai import tsai_2D

from mod4 import setup
FRAMES = 200


# integration & physical parameters
integration_params = dict(  dt=1e-3, n_steps=30, 
                            Lx=10, Lv=10, dx=0.1, dv=0.1, 
                            ADI=False,
                            diffCN=True,
                            CN=np.array([True, True, True]))

physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)
X, V = get_quad_mesh(integration_params)

# Initial conditions
x0, v0 = 0,0
t0 = 10
p0 = analytic(X,V, t0, x0, v0, physical_params)

# p0 = ((X**2 + V**2) <1).astype(float)
# p0 = np.ones((len(x), len(v)))
# r = np.sqrt(X**2 + V**2)
# p0 = np.exp( - ((r-0.5)/0.2)**2)
p0 = np.exp(-((X)**2 + (V-2)**2)/0.5**2)

p_num = np.real(p0)
p_num /= quad_int(p_num , integration_params)
p_an = p0

# What to plot
preproc_func = lambda x: np.log(np.abs(x))
levels = np.linspace(-40,1, 30)

# Definition of the plots
fig, axes = plt.subplot_mosaic([["imag"], ["sect"]], height_ratios=[1, 0.2], constrained_layout=True)

axes, ax2 = axes.values()
axes.set_aspect('equal')

n_plot = axes.contourf(X, V, preproc_func(p_num), levels=levels, cmap='rainbow')


P = np.zeros((p_num.shape[0], p_num.shape[1]))


sections = dict(x=[dict(coord=55)], 
                v=[dict(coord=70)]
                )

for c in sections['x']:
    c['plot'], = ax2.plot(X[ c['coord'],:],p_num[:, c['coord']])

for c in sections['v']:
    c['plot'], = ax2.plot(V[:,c['coord']],p_num[c['coord'],:])

# ax2.set_ylim(1e-20,1)
# ax2.set_yscale('log')

for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        P[i,j] = p_num[i,j]
        c=1
        if i != P.shape[0]-1:
            P[i,j] += p_num[i+1, j]
            c+=1
        if j != P.shape[1] - 1:
            P[i,j] += p_num[i, j+1]
            c+=1
        if i!=P.shape[0]-1 and j != P.shape[1]-1:
            P[i,j] += p_num[i+1, j+1]
            c+=1
        P[i,j] /= c

mappable=axes.pcolormesh(X, V, preproc_func(p_num), cmap='rainbow', vmin=np.min(levels), vmax = np.max(levels))
plt.colorbar(mappable, ax=axes, label="log(abs(p))")

def update(i):
    if i == 0:
        return
    global p_num, p_an , P 

    t = t0 + i*integration_params['n_steps']*integration_params['dt']
    # Sets the simulation to start at the last time 
    # Useful only for time-dependent potentials
    physical_params['t0'] = i*integration_params['n_steps']*integration_params['dt']

    # Numeric
    # p_num , norm , curr = generic_3_step(p_num, physical_params, integration_params, save_norm=True)
    p_num, P = tsai_2D(p_num, P, physical_params, integration_params)
    p_num = np.array(p_num)
    # p_num[p_num<0] = 0.0

    print("norm", quad_int(p_num, integration_params))
    # p_num /= norm[-1]

    axes.clear()
    mappable=axes.pcolormesh(X, V, preproc_func(p_num), cmap='rainbow', vmin=np.min(levels), vmax = np.max(levels))
    axes.contour(X, V, p_num, colors=["r"] + 9*['k'], levels=np.linspace(0, 1, 10))

    fig.suptitle(f"IMPLICIT t = {t:4.3f}, norm={quad_int(p_num, integration_params):.2f}")

    for c in sections['x']:
        axes.plot(X[:, c['coord']], V[:, c['coord']])
        c['plot'].set_data(X[ c['coord'],:], p_num[:, c['coord']])

    for c in sections['v']:
        axes.plot(X[c['coord'],:], V[c['coord'],:])
        c['plot'].set_data(V[:,c['coord']], p_num[c['coord'],:])
    return  


anim = FuncAnimation(fig, update, frames=FRAMES, interval=3/60*1e3, blit=False,)
# anim.save("arm_osc_impl.mp4")
plt.show()
