import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mod4.utils import analytic
from mod4.implicit import  generic_3_step as funker_plank
from mod4.utils import quad_int, get_quad_mesh

FRAMES = 300


# integration & physical parameters
integration_params = dict(  dt=1/1000, n_steps=300, 
                            Lx=10, Lv=10, dx=0.1, dv=0.1, 
                            ADI=False,
                            diffCN=True,
                            CN=np.array([True, True, True]))

physical_params = dict(omega_squared=1.0, gamma=0.05, sigma_squared=0.001**2)
X, V = get_quad_mesh(integration_params)

# Initial conditions
x0, v0 = 1,0
t0 = .95
p0 = analytic(X,V, t0, x0, v0, physical_params)

# p0 = ((X**2 + V**2) <1).astype(float)
# p0 = np.ones((len(x), len(v)))
r = np.sqrt(X**2 + V**2)
p0 = np.exp( - ((r-1)/0.5)**2)
p_num = np.real(p0)
p_num /= quad_int(p_num , integration_params)
p_an = p0

# What to plot
preproc_func = lambda x: np.abs(x)
levels = np.linspace(0,0.3, 30)

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
axes['covar'].axis('off')

def update(i):
    if i == 0:
        return
    global p_num, p_an    

    t = t0 + i*integration_params['n_steps']*integration_params['dt']
    # Sets the simulation to start at the last time 
    # Useful only for time-dependent potentials
    physical_params['t0'] = i*integration_params['n_steps']*integration_params['dt']

    # Numeric
    p_num , norm , curr = funker_plank(p_num, physical_params, integration_params, save_norm=True)
    p_num = np.array(p_num)
    p_num[p_num<0] = 0
    # p_num /= norm[-1]

    # Analytic
    p_an = np.real(analytic(X,V, t , x0, v0, physical_params))
    # print(quad_int(np.real(p_an), x,v))
    # Moments
    mu_num.append(quad_int(X*np.real(p_num), integration_params))
    mu_an.append(quad_int(X*np.real(p_an), integration_params))

    sigma_an.append(quad_int((X-mu_an[-1])**2*np.real(p_an), integration_params))
    sigma_num.append(quad_int((X-mu_num[-1])**2*np.real(p_num), integration_params))
    print(np.sqrt(np.mean((np.array(p_num) - p_an)**2)))
    axes['mean'].clear()
    axes['mean'].plot(np.linspace(t0, t, len(mu_num)),mu_num, label="numeric")
    axes['mean'].plot(np.linspace(t0, t, len(mu_an)), mu_an, label="analytic")
    axes['mean'].set_xlim(0, FRAMES*integration_params['dt']*integration_params['n_steps'])
    axes['mean'].set_ylim(-2,3)

    axes['var'].clear()
    axes['var'].plot(np.linspace(t0, t, len(sigma_num)),sigma_num, label="numeric")
    axes['var'].plot(np.linspace(t0, t, len(sigma_an)), sigma_an, label="analytic")
    axes['var'].set_xlim(0, FRAMES*integration_params['dt']*integration_params['n_steps'])
    axes['var'].set_ylim(0,physical_params['sigma_squared']/physical_params['gamma'])
    axes['var'].axhline(0.5*physical_params['sigma_squared']/physical_params['gamma'], ls=":", color='k')

    for axname, p_to_plot in zip(['analytic', 'numeric'], [p_an, p_num]):
        axes[axname].clear()
        axes[axname].contourf(X, V, preproc_func(p_to_plot), levels=levels, cmap='rainbow')
        axes[axname].set_title(axname)
    fig.suptitle(f"t = {t:.3}")
    axes['mean'].legend()
    axes['var'].legend()

    # print(i, end="-", flush=True)
    return  


anim = FuncAnimation(fig, update, frames=FRAMES, interval=3/60*1e3, blit=False,)
# anim.save("anim_comparison_underdamped.mp4")
plt.show()
