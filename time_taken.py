from mod4.diffeq import generic_3_step
from mod4.utils import get_quad_mesh, analytic
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
from time import perf_counter

import matplotlib.pyplot as plt

physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)

N_grid_points = 10
N_time_points = 10

nsteps= (200/N_time_points*np.arange(1, N_time_points+1)).astype(int)
grid_dim = (200/N_grid_points*np.arange(1, N_grid_points+1)).astype(int)

print(nsteps, grid_dim)
times = np.zeros((N_time_points, N_grid_points))

for i, n_steps in enumerate(nsteps):

    for j, N in enumerate(grid_dim):
        print(i/len(nsteps)*100,"-",j/len(grid_dim)*100)
        # integration & physical parameters
        integration_params = dict(  dt=3.14/1000, n_steps=n_steps, 
                                    Lx=8, Lv=8, dx=8/N, dv=8/N, 
                                    ADI=False, 
                                    CN=np.array([True, True, True]))
        X, V = get_quad_mesh(integration_params)

        # Initial conditions
        x0, v0 = 1,0
        t0 = .95
        
        p0 = np.exp( -((X-x0)/2)**2 - ((V-v0)/2)**2)


        start = perf_counter()
        p , norm, curr = generic_3_step(p0, physical_params, integration_params, save_norm=False)
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