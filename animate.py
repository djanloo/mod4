
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mod4.utils import analytic
from mod4.implicit import  generic_3_step
from mod4.utils import quad_int, get_quad_mesh, get_lin_mesh
from mod4.tsai import tsai_2D_leapfrog

# from mod4 import setup
FRAMES = 400


# integration & physical parameters
integration_params = dict(  dt=3e-3, n_steps=10, 
                            Lx=10, Lv=10, dx=0.1, dv=0.1, 
                            ADI=False,
                            diffCN=True,
                            CN=np.array([True, True, True]))

physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.01**2)
X, V = get_quad_mesh(integration_params)

# Initial conditions
x0, v0 = -1,-1
t0 = 0.001
p0 = analytic(X,V, t0, x0, v0, physical_params)

# p0 = ((X**2 + V**2) <2).astype(float)

r = np.sqrt(X**2 + V**2)
p0 = np.exp( - ((r-2)/0.5)**2)
# p0 = np.exp(-((X-x0)**2 + (V- v0)**2)/0.5**2)

p_num = np.real(p0)
p_num /= quad_int(p_num , integration_params)
p_an = p0

# What to plot
preproc_func = lambda x: x
levels = preproc_func(np.linspace(1e-40 , 0.1, 30))

# Definition of the plots
fig, axes = plt.subplot_mosaic([['avgx','avgv', "imag"], ['sect', "sect", 'sect']], height_ratios=[1, 0.2], constrained_layout=True, figsize=(10,5))

axavgx,axavgv, axes, ax2 = axes.values()
axes.set_aspect('equal')
axavgv.set_aspect('equal')
axavgx.set_aspect('equal')

n_plot = axes.contourf(X, V, preproc_func(p_num), levels=levels, cmap='rainbow')


sections = dict(x=[dict(coord=p0.shape[1]//2)], 
                v=[dict(coord=p0.shape[0]//2)]
                )
ax2.set_ylim(0,1)

for c in sections['x']:
    c['plot'], = ax2.plot(X[ c['coord'],:],p_num[:, c['coord']])

for c in sections['v']:
    c['plot'], = ax2.plot(V[:,c['coord']],p_num[c['coord'],:])


Px = np.zeros((p_num.shape[0], p_num.shape[1]-1))
Pv = np.zeros((p_num.shape[0]-1, p_num.shape[1]))

for j in range(Px.shape[0]):
    for i in range(Px.shape[1]):
        Px[j,i] = 0.5*(p0[j, i] + p0[j, i+1])

for j in range(Pv.shape[0]):
    for i in range(Pv.shape[1]):
        Pv[j,i] = 0.5*(p0[j, i] + p0[j+1, i])

mappable=axes.pcolormesh(X, V, preproc_func(p_num), cmap='rainbow', vmin=np.min(levels), vmax = np.max(levels))

plt.colorbar(mappable, ax=axes, label="log(abs(p))")

def update(i):
    if i == 0:
        return
    global p_num, p_an , Px, Pv

    t = t0 + i*integration_params['n_steps']*integration_params['dt']
    # Sets the simulation to start at the last time 
    # Useful only for time-dependent potentials
    physical_params['t0'] = i*integration_params['n_steps']*integration_params['dt']

    # Numeric
    p_num , norm , curr = generic_3_step(p_num, physical_params, integration_params, save_norm=True)
    switch = i#(i//20)%2
    print(f'switch {switch}')
    p_num, Px, Pv = tsai_2D_leapfrog(p_num, Px, Pv, physical_params, integration_params, switch=switch)
    # p_num , norm , curr = generic_3_step(p_num, physical_params, integration_params, save_norm=True)
    p_num = np.array(p_num)
    # p_num[p_num<0] = 0.0

    # print("norm", quad_int(p_num, integration_params))
    # p_num /= norm[-1]

    axes.clear()
    axavgx.clear()
    axavgv.clear()

    mappable=axes.pcolormesh(X, V, preproc_func(p_num), cmap='rainbow', vmin=np.min(levels), vmax = np.max(levels))
    axavgx.pcolormesh(preproc_func(Px), cmap='rainbow', vmin=np.min(levels), vmax = np.max(levels))
    axavgv.pcolormesh(preproc_func(Pv), cmap='rainbow', vmin=np.min(levels), vmax = np.max(levels))

    axavgx.contour(preproc_func(Px), colors=["r"] + 9*['k'], levels=np.linspace(0, 1, 10))
    axavgv.contour(preproc_func(Pv), colors=["r"] + 9*['k'], levels=np.linspace(0, 1, 10))

    axes.contour(X, V, p_num, colors=['k'], levels=levels[::3]+1e-5)

    fig.suptitle(f"evolving: {'X' if switch==0 else 'V'}\nt = {t:4.3f}, norm={quad_int(p_num, integration_params):.2f}")
    axavgx.set_title("AVG - X")
    axavgv.set_title("AVG - V")

    for c in sections['x']:
        axes.plot(X[:, c['coord']], V[:, c['coord']])
        c['plot'].set_data(X[ c['coord'],:], p_num[:, c['coord']])
        # ax2.set_ylim(np.min(p_num[:, c['coord']]), np.max(p_num[:, c['coord']]))

    for c in sections['v']:
        axes.plot(X[c['coord'],:], V[c['coord'],:])
        c['plot'].set_data(V[:,c['coord']], p_num[c['coord'],:])
    return  


anim = FuncAnimation(fig, update, frames=FRAMES, interval=3/60*1e3, blit=False,)
# anim.save("weird_symmetry4", writer='ffmpeg')
plt.show()
