import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns;
from mod4.diffeq import FTCS, LAX, LAX_WENDROFF, burgers_lw, heat_cn, diff_advec

N = 100
Ntimes = 3
x = np.linspace(0, 1, N, endpoint=False)

# u0 = np.sin(5*2*np.pi*x) + 0.5*np.sin(6*2*np.pi*x) + 0.2*np.sin(20*2*np.pi*x) + 0.5*np.sin(30*2*np.pi*x) 
u0 = np.exp(- ((x-0.5)/0.1)**6)
dx = x[1]-x[0]

integration_args = dict(dx=dx, dt=1e-4, n_steps=1e1, nu=0.8, c=30.0)

fig, axx = plt.subplots()
colors = sns.color_palette("flare", Ntimes)

axx.plot(x, u0, color='b', marker='.')

for i in range(Ntimes):
    integration_args["dt"] /= 2
    integration_args["n_steps"] *= 2
    u = diff_advec(u0, **integration_args)
    axx.plot(x, u, color=colors[i])
plt.show()