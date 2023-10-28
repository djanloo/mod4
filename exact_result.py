import numpy as np
import matplotlib.pyplot as plt
from mod4.utils import analytic, quad_int
from mod4.diffeq import funker_plank

Lx, Lv = 4, 4
x0, v0 = 2,  0.0

t0 = 0.1
t1 = 20

physical_params = dict(omega_squared=1, gamma=1.0, sigma= 0.8)
integration_params = dict(dt=0.1, n_steps=(1*int((t1 - t0)/0.1)))

print(integration_params)
x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)

p0 = analytic(X, V, t0, x0, v0, physical_params)
print("innit norm",quad_int(np.real(p0), x, v))
print("max abs init",np.max(np.abs(p0)))
plt.contourf(np.real(p0))
plt.show()

fig, axes = plt.subplot_mosaic([['exact', 'cbar', 'numeric']], width_ratios=[1, 0.1, 1], constrained_layout=True)
levels = np.linspace(0,0.5, 30)

p_an =  np.real(analytic(X,V, t1, x0, v0, physical_params))
mappable = axes['exact'].contourf(X, V,p_an, levels=levels)

p_num, norm = funker_plank(np.real(p0), x, v, physical_params, integration_params)
axes['numeric'].contourf(X, V, p_num, levels=levels)

plt.colorbar(mappable, cax=axes['cbar'])

axes['exact'].set_aspect('equal')
axes['numeric'].set_aspect('equal')

axes['numeric'].set_title("numeric")
axes['exact'].set_title('exact')
plt.show()