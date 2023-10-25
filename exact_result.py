import numpy as np
import matplotlib.pyplot as plt
from mod4.diffeq import analytic
from mod4.diffeq import funker_plank

Lx, Lv = 4, 4
x0, v0 = 2,  0.0

t0 = 0.01
t1 = 0.5

physical_params = dict(omega_squared=1, gamma=2.1, sigma= 0.8)
integration_params = dict(dt=0.1, n_steps=(10*int((t1 - t0)/0.1)))

print(integration_params)
x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)

p0 = analytic(X, V, t0, x0, v0, physical_params)


fig, axes = plt.subplot_mosaic([['exact', 'cbar', 'numeric']], width_ratios=[1, 0.1, 1], constrained_layout=True)
mappable = axes['exact'].contourf(X, V, analytic(X,V, t1, x0, v0, physical_params))

p, norm = funker_plank(p0, x, v, physical_params, integration_params)
axes['numeric'].contourf(X, V, p)

plt.colorbar(mappable, cax=axes['cbar'])

axes['exact'].set_aspect('equal')
axes['numeric'].set_aspect('equal')

axes['numeric'].set_title("numeric")
axes['exact'].set_title('exact')
plt.show()