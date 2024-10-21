# -*- coding: utf-8 -*-
"""corte2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VTU5w5pyMi_O5w1_jJvFiBdU6Hh0IPg4
"""

# Trabajo presentado por Santiago Carvajal del salón TS7A en la materia Ciencia de Datos
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


datos = sns.load_dataset('titanic')

print("Información general del conjunto de datos:")
datos.info()

print("\nPrimeras filas del conjunto de datos:")
print(datos.head())

print("\nEstadísticas descriptivas:")
print(datos.describe())

datos.dropna(subset=['age', 'embarked', 'sex', 'pclass', 'survived'], inplace=True)

plt.figure(figsize=(8, 6))
sns.barplot(x="pclass", y="survived", data=datos, ci=None)
plt.title('Tasa de Supervivencia por Clase')
plt.ylabel('Tasa de Supervivencia')
plt.xlabel('Clase del Pasajero')
plt.xticks(ticks=[0, 1, 2], labels=['Primera', 'Segunda', 'Tercera'])
plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(x="survived", y="age", data=datos, palette="muted")
plt.title('Distribución de la Edad por Supervivencia')
plt.xlabel('Supervivencia (0 = No, 1 = Sí)')
plt.ylabel('Edad')
plt.xticks(ticks=[0, 1], labels=['No', 'Sí'])
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x="sex", y="survived", data=datos, ci=None)
plt.title('Tasa de Supervivencia por Género')
plt.ylabel('Tasa de Supervivencia')
plt.xlabel('Género')
plt.xticks(ticks=[0, 1], labels=['Hombre', 'Mujer'])
plt.show()

datos_numericos = datos.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 8))
correlacion = datos_numericos.corr()
sns.heatmap(correlacion, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación de las Variables Numéricas del Titanic')
plt.show()


print("\nInterpretación de los resultados:")
print("- Los pasajeros de primera clase tienen una mayor probabilidad de sobrevivir.")
print("- Los pasajeros más jóvenes tienden a tener una mayor tasa de supervivencia.")
print("- Las mujeres tienen una tasa de supervivencia considerablemente mayor que los hombres.")
