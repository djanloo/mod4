import numpy as np
from matplotlib import pyplot as plt

from mod4.diffeq import FTCS, LAX, LAX_WENDROFF

x = np.linspace(-10, 10, 300)
u0 = np.exp(-(x/0.5)**2)
v = 30
intgration_args = dict(dx=x[1]-x[0], dt=0.000001, v=v, n_steps=1_000)


plt.plot(u0)

u = u0.copy()
for i in range(4):
    u = LAX_WENDROFF(u, **intgration_args)
    plt.plot(u)

plt.show()