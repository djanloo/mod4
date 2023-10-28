import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mod4.utils import analytic
from mod4.diffeq import funker_plank
from mod4.utils import quad_int

FRAMES = 300

# Environment setting
Lx, Lv = 4, 4
x, v = np.linspace(-Lx ,Lx, 100, endpoint=False), np.linspace(-Lv, Lv, 100, endpoint=False)
X, V = np.meshgrid(x,v)
t0 = .95

# integration & physical parameters
integration_params = dict(dt=np.pi/1000, n_steps=20)
physical_params = dict(omega_squared=1.0, gamma=2.1, sigma=0.8)

# Initial conditions
x0, v0 = 0,3
sx, sv = 0.2,  0.2
p0 = analytic(X,V, t0, x0, v0, physical_params)

p_num = np.real(p0)
p_an = p0

# What to plot
preproc_func = lambda x: np.abs(x)
levels = np.linspace(0,1.5, 30)

mu_num = []
mu_an = []
sigma_an = []
sigma_num = []

# Definition of the plots
fig, axes = plt.subplot_mosaic([["analytic", 'cbar', 'numeric'], ['mean', 'covar', 'var']], width_ratios=[1, 0.1, 1], constrained_layout=True)

n_plot = axes['numeric'].contourf(X, V, preproc_func(p_num), levels=levels, cmap='rainbow')
a_plot = axes['analytic'].contourf(X, V, preproc_func(p_an), levels=levels, cmap='rainbow')
cbar = fig.colorbar(a_plot, cax=axes['cbar'])

mu_n_plot, = axes['mean'].plot([0, 1], [0, 0], label="numeric")
mu_an_plot, = axes['mean'].plot([0,1],[0,0], label="analytic")
sigma_an_plot = axes['var'].plot([0,1],[0,0], label="analytic")
sigma_num_plot = axes['var'].plot([0,1],[0,0], label="numeric")


def update(i):
    global p_num, p_an    
    t = t0 + (i+1)*integration_params['n_steps']*integration_params['dt']

    # Sets the simulation to start at the last time 
    # Useful only for time-dependent potentials
    physical_params['t0'] = i*integration_params['n_steps']*integration_params['dt']

    # Numeric
    p_num , norm = funker_plank(p_num, x, v, physical_params, integration_params, save_norm=True)
    p_num = np.array(p_num)
    p_num[p_num<0] = 0
    p_num /= norm[-1]

    # Analytic
    p_an = analytic(X,V, t , x0, v0, physical_params)

    # Moments
    mu_num.append(quad_int(X*np.real(p_num), x, v))
    mu_an.append(quad_int(X*np.real(p_an), x, v))

    sigma_an.append(quad_int((X-mu_an[-1])**2*np.real(p_an), x,v))
    sigma_num.append(quad_int((X-mu_num[-1])**2*np.real(p_num), x,v))

    axes['mean'].clear()
    axes['mean'].plot(np.linspace(t0, t, len(mu_num)),mu_num, label="numeric")
    axes['mean'].plot(np.linspace(t0, t, len(mu_an)), mu_an, label="analytic")
    axes['mean'].set_xlim(0, FRAMES*integration_params['dt']*integration_params['n_steps'])
    axes['mean'].set_ylim(-2,2)

    axes['var'].clear()
    axes['var'].plot(np.linspace(t0, t, len(sigma_num)),sigma_num, label="numeric")
    axes['var'].plot(np.linspace(t0, t, len(sigma_an)), sigma_an, label="analytic")
    axes['var'].set_xlim(0, FRAMES*integration_params['dt']*integration_params['n_steps'])
    axes['var'].set_ylim(0,1)

    for axname, p_to_plot in zip(['analytic', 'numeric'], [p_an, p_num]):
        axes[axname].clear()
        axes[axname].contourf(X, V, preproc_func(p_to_plot), levels=levels, cmap='rainbow')
        axes[axname].set_title(axname)
    fig.suptitle(f"t = {t:.3}")
    axes['mean'].legend()
    axes['var'].legend()

    print(i, end="-", flush=True)
    return  


anim = FuncAnimation(fig, update, frames=FRAMES, interval=3/60*1e3, blit=False)
# anim.save("anim_comparison.mp4")
plt.show()
