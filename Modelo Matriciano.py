import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.arange( 0 , 5 , 0.1), np.arange( 0 , 3 , 0.1))
X1 = x.ravel()
X2 = y.ravel()
Y=2*X1 + 1.5*X2 + 3 + np.random.randn(*X1.shape) * 0.3

unos = np.ones_like(X1)
X = np.column_stack((unos, X1, X2))
coeficientes = np.linalg.inv(X.T @ X) @ (X.T @ Y)
print(coeficientes)

Y_predict = coeficientes[0] + coeficientes[1]*X1 + coeficientes[2]*X2

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot(X1, X2, Y,'green',label='Modelo original')
ax.set_xlim(X1.min(), X1.max())  
ax.set_ylim(X2.min(), X2.max()) 
ax.set_zlim(Y.min(), Y.max()) 
plt.legend()
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(X1, X2, Y_predict,'red', label='Modelo matriciano')
ax2.set_xlim(X1.min(), X1.max())  
ax2.set_ylim(X2.min(), X2.max()) 
ax2.set_zlim(Y.min(), Y.max()) 
plt.suptitle(f"Y = {round(coeficientes[1],2)}X1 + {round(coeficientes[2],2)}X2 + {round(coeficientes[0],2)}")
plt.legend()
plt.show()

