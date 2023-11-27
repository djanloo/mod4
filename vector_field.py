import numpy as np
import matplotlib.pyplot as plt

# Genera dati per il campo vettoriale
x = np.linspace(-3, 3, 15)
v = np.linspace(-3, 3, 15)
X, V = np.meshgrid(x, v)

U_dot = V  # Componente x del vettore
V_dot = - X - V # Componente y del vettore

# Crea la figura e gli assi
fig, ax = plt.subplots()

# Plotta il campo vettoriale
ax.quiver(X,V,  U_dot, V_dot, scale=50)

# Imposta le etichette degli assi
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Aggiunge la griglia
# ax.grid(True)

# Mostra il grafico
plt.show()
plt.quiver