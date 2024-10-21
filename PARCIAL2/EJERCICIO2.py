# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KLGeVr5udBp0-fqW5N1IOEf4fsmgmOWb
"""

# ITrabajo presentado por Santiago Carvajal del salon TS7A
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


iris = sns.load_dataset('iris')


plt.figure(figsize=(10, 6))
plt.scatter(iris['sepal_length'], iris['sepal_width'], c=iris['species'].astype('category').cat.codes, cmap='viridis', edgecolor='k', s=100)
plt.title('Gráfico de Dispersión: Longitud vs. Anchura del Sépalo')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Anchura del Sépalo')
plt.colorbar(label='Especie')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(iris['sepal_length'], kde=True, bins=30)
plt.title('Distribución de la Longitud del Sépalo')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Frecuencia')
plt.show()


sns.pairplot(iris, hue='species')
plt.suptitle('Gráfico de Pares del Dataset de Iris', y=1.02)
plt.show()


plt.figure(figsize=(10, 6))
correlation = iris.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de Correlación')
plt.show()