from IPython.display import clear_output
import random
import numpy as np
import matplotlib.pyplot as plt
X_real = np.linspace(0, 4, 100)
Y_real = 2 * X_real + 3 + np.random.randn(*X_real.shape) * 0.3
# Como es un modelo lineal, la ecuación modelo es Y=m*X + b, así:
#Valores iniciales
m = random.randint(0, 5)
b = random.randint(0, 5)

# Opciones adicionales..
display_step = 50 #Es un comodín para decidir cada cuantas iteraciones deseamos ver los resultados.
N = len(X_real)
loss = [] #Pèrdida

# Proceso de optimización
epochs = 500
learning_rate = 0.01
convergence_criteria = 1e-5 #0.00001
for step in range( epochs ):
    # Gradientes iniciales (Zero gradients)
    m_gradient = 0
    b_gradient = 0
    error = 0
    Diferencia = 0

    # Calculamos el error y con ello, el gradiente
    for i in range(N):
      Yp = (m*X_real[i] + b)
      Diferencia = Yp - Y_real[i]
      error += (Diferencia)**2 #MSE
      # Calculamos el gradiente
      m_gradient += (2/N)*(Diferencia)*X_real[i]
      b_gradient += (2/N)*(Diferencia)*1

    MSE = error/N
    
    # Actualizamos los valores de acuerdo al gradiente
    m = m - ( m_gradient * learning_rate )
    b = b - ( b_gradient * learning_rate )
    
    # Guardamos el objetivo por cada epoca
    loss.append(np.abs(MSE))
    
    # Mostramos cada display_step epocas
    if step % display_step == 0:
        plt.get_current_fig_manager().window.state('zoomed')  # Maximiza la ventana del gráfico en la pantalla
        plt.clf()
        # gráfico de la recta de regresión
        plt.subplot(1, 2, 1)
        plt.scatter(X_real, Y_real, color='b')
        pred_x = [0, max(X_real)]
        pred_y = [m*0+b, m*max(X_real) + b]
        plt.title(f'Epoca: {step}, MSE: {loss[-1]}')
        plt.plot(pred_x, pred_y, "r")
        # gráfico del costo (MSE)
        plt.subplot(1, 2, 2)
        plt.plot(loss)
        plt.title('Costo (MSE)')
        plt.xlabel('Epocas')
        plt.ylabel('MSE')
        plt.tight_layout()
        plt.pause(0.1)      
        clear_output()

    #Paramos el algoritmo cuando los hiperparámetros no estén cambiando lo suficiente
    if max(abs(learning_rate * m_gradient), abs(learning_rate * b_gradient)) < convergence_criteria:
        break
plt.close()

#=============Impresión de los resultados==========
print("Los valores obtenidos son: ", 'm=',m, 'y b=', b)
print("Finalizado en ", step, " iteraciones ")
print('')
Y_prediction4=m * X_real + b
MSE=np.mean(np.square(Y_prediction4 - Y_real))
print('MSE= ',MSE)

_, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].scatter(X_real, Y_real, c="b"),axes[0].set_title("Datos & Modelo")
axes[0].plot(X_real,Y_prediction4, c="r")
axes[1].plot( loss ), axes[1].set_title("Costo (MSE)")
axes[1].set_xlabel("Epochs"), axes[1].set_ylabel("MSE")
plt.get_current_fig_manager().window.state('zoomed')  # Maximiza la ventana del gráfico en la pantalla
plt.show()
