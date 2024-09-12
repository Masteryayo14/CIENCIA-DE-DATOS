print ("Nombre: Santiago Carvajal")
print ("Materia: Ciencia de Datos")
print ("Salón: TS7A")
print ("SEGUNDO PUNTO")

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)

X = iris.data.features  
y = iris.data.targets  


print("")
print("Características (X):")
print(X.head())  

print("\nObjetivo (y):")
print(y.head())  

# Calcular estadísticas descriptivas para las características
print("\nEstadísticas Descriptivas de las Características:")
print("Media:")
print(X.mean())  # Media

print("\nMediana:")
print(X.median())  # Mediana

print("\nDesviación Estándar:")
print(X.std())  # Desviación estándar
