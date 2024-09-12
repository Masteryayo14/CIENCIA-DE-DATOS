print ("Nombre: Santiago Carvajal")
print ("Materia: Ciencia de Datos")
print ("Salón: TS7A")
print ("TERCER PUNTO")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Cargar el dataset de Boston Housing desde openml
boston = fetch_openml(name='boston', version=1, as_frame=True)

# Datos de características (X) y objetivo (y)
X = boston.data  # Variables independientes
y = boston.target.values  # Convertir y a un array de numpy

# Seleccionamos la característica 'RM' (número de habitaciones) para la regresión
X_rm = X['RM'].values.reshape(-1, 1)  # Reformateamos los datos para el modelo

# Inicialización de parámetros
m = 0  # Pendiente
b = 0  # Intercepto
L = 0.01  # Tasa de aprendizaje
epochs = 1000  # Número de iteraciones

n = float(len(X_rm))  # Número de datos

# Implementación del algoritmo de gradiente descendente
for i in range(epochs):
    y_pred = m * X_rm + b  # Predicción actual
    error = y - y_pred.flatten()  # Error actual, aplanar y_pred para que tenga la misma dimensión que y
    D_m = (-2/n) * np.dot(X_rm.T, error)  # Derivada parcial con respecto a m
    D_b = (-2/n) * np.sum(error)  # Derivada parcial con respecto a b
    m = m - L * D_m  # Actualización de m
    b = b - L * D_b  # Actualización de b

# Predicción final con los parámetros ajustados
y_pred = m * X_rm + b

# Visualización de la regresión lineal
plt.scatter(X_rm, y, color='purple')  # Gráfico de dispersión de los datos reales
plt.plot(X_rm, y_pred, color='Black')  # Línea de regresión
plt.xlabel('Número de habitaciones (RM)')
plt.ylabel('Valor medio de las casas (MEDV)')
plt.title('Regresión Lineal de Boston Housing')
plt.show()
