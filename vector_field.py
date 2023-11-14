import numpy as np
import matplotlib.pyplot as plt

# Genera dati per il campo vettoriale
x = np.linspace(-3, 3, 15)
y = np.linspace(-3, 3, 15)
X, Y = np.meshgrid(x, y)
U =  Y  # Componente x del vettore
V = -X - 0.5*Y*(X**2 - 1)  # Componente y del vettore

# Crea la figura e gli assi
fig, ax = plt.subplots()

# Plotta il campo vettoriale
ax.quiver(X, Y, U, V, scale=50)

# Imposta le etichette degli assi
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Aggiunge la griglia
# ax.grid(True)

# Mostra il grafico
plt.show()
