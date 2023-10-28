from mod4.diffeq import funker_plank
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
from time import perf_counter

import matplotlib.pyplot as plt

Lx, Lv = 4, 4
x0, v0 = 0.0,  0.0
sx, sv = 0.6,  0.6
x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)
p0 = np.exp( -((X-x0)/sx)**2 - ((V-v0)/sv)**2)
p0 /= np.sum(p0)*np.diff(x)[0]*np.diff(v)[0]

physical_params = dict(omega_squared=1.0, gamma=2.1, sigma= 0.8)

N_grid_points = 10
N_time_points = 10

nsteps= (200/N_time_points*np.arange(1, N_time_points+1)).astype(int)
grid_dim = (200/N_grid_points*np.arange(1, N_grid_points+1)).astype(int)

print(nsteps, grid_dim)
times = np.zeros((N_time_points, N_grid_points))

for i, n_steps in enumerate(nsteps):

    for j, N in enumerate(grid_dim):

        integration_params = dict(dt=np.pi/1000.0, n_steps=n_steps)
        
        x, v = np.linspace(-Lx ,Lx, N, endpoint=False), np.linspace(-Lv, Lv, N, endpoint=False)
        X, V = np.meshgrid(x,v)
        p0 = np.exp( -((X-x0)/sx)**2 - ((V-v0)/sv)**2)
        p0 /= np.sum(p0)*np.diff(x)[0]*np.diff(v)[0]

        start = perf_counter()
        p , norm = funker_plank(p0, x, v, physical_params, integration_params, save_norm=False)
        times[i,j] = (perf_counter() - start)*1e6 # conversione in us


times_N = np.zeros(N_grid_points)
sigmas = np.zeros(N_grid_points)

for j in range(len(grid_dim)):
    times_N[j] = np.mean(times[:, j]/nsteps)
    sigmas[j] = np.std(times[:, j]/nsteps)

print(times_N.shape, sigmas.shape)
plt.errorbar(grid_dim, times_N, sigmas,ls="", marker=".")

def cost(N, beta, gamma):
    return beta*N**2 + gamma*N

inital_guess = (0.4, 1e-1)
# plt.plot(grid_dim, cost(grid_dim, *inital_guess))

from scipy.optimize import curve_fit
pars, popt = curve_fit(cost, grid_dim, times_N, sigma=sigmas, p0=inital_guess)
plt.plot(grid_dim, cost(grid_dim, *pars))
print(pars)

plt.show()