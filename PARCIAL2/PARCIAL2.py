# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MNlsYL-dokYJP_aQmqQ7kEHFsZbELHYB
"""

!pip install kaggle
!pip install seaborn

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d aungpyaeap/supermarket-sales

!unzip supermarket-sales.zip

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('supermarket_sales - Sheet1.csv')

df.head()

sales_by_product = df.groupby('Product line')['Total'].sum()

plt.figure(figsize=(10, 6))
plt.bar(sales_by_product.index, sales_by_product.values, color='lightcoral')
plt.title('Ventas Totales por Categoría de Producto', fontsize=16)
plt.xlabel('Categoría de Producto', fontsize=12)
plt.ylabel('Monto Total de Ventas', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(sales_by_product.index, sales_by_product.values, color='lightgreen')
ax.set_title('Ventas Totales por Categoría de Producto (Enfoque OO)', fontsize=16)
ax.set_xlabel('Categoría de Producto', fontsize=12)
ax.set_ylabel('Monto Total de Ventas', fontsize=12)
ax.set_xticklabels(sales_by_product.index, rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Total'], kde=True, color='dodgerblue')
plt.title('Distribución del Monto Total de Ventas', fontsize=16)
plt.xlabel('Monto Total de Ventas', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='Total', data=df, color='purple', s=50, alpha=0.6)
plt.title('Relación entre Cantidad de Productos Vendidos y Monto Total', fontsize=16)
plt.xlabel('Cantidad de Productos Vendidos', fontsize=12)
plt.ylabel('Monto Total de Ventas', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 6))
sns.catplot(x='City', y='Total', hue='Product line', data=df, kind='bar', height=6, aspect=1.5, palette='coolwarm')
plt.title('Ventas por Ciudad y Categoría de Producto', fontsize=16)
plt.xlabel('Ciudad', fontsize=12)
plt.ylabel('Monto Total de Ventas', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

custom_palette = sns.color_palette("gist_rainbow")

sns.catplot(x='City', y='Total', hue='Product line', data=df, kind='bar', palette=custom_palette, height=6, aspect=1.5)
plt.title('Ventas por Ciudad y Categoría de Producto (Paleta Personalizada)', fontsize=16)
plt.xlabel('Ciudad', fontsize=12)
plt.ylabel('Monto Total de Ventas', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()