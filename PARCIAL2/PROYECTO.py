#####################################################################################
# AUTOR: BRAYAN MATALLANA JOYA /SANTIAGO CARVAJAL / MATHEW ESPINOSA                 #
# FECHA: 29/10/2024                                                                 #
# DESCRIPCION: PROYECTO, Distribucion de Medicamentos en Zonas Rurales CORTE 2      #
# ASIGNATURA: BIG DATA, CIENCIA DE DATOS- TS7A                                      #
#####################################################################################

# ZONA DE IMPORTACIONES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# se debe instalar pip install openpyxl   para que lea archivos excel.

# Cargar el archivo Excel
df = pd.read_excel(r'C:\Users\ASUS\Desktop\U\CIENCIA_DE_DATOS_BIG_DATA\CORTE_2\CORTE_2\distribucion_medicamentos_rurales.xlsx') #esta ruta cambia dependiendo donde se encuentre el excel

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 1. Histograma de la Demanda de Medicamentos
plt.subplot(2, 2, 1)
sns.histplot(df['Demanda_Medicamentos'], bins=10, color='skyblue')
plt.title('Distribucion de la Demanda de Medicamentos')
plt.xlabel('Demanda de Medicamentos')
plt.ylabel('Frecuencia')

# 2. Grafico de Barras - Refrigeracion Disponible por Comunidad
plt.subplot(2, 2, 2)
sns.countplot(data=df, x='Refrigeracion_Disponible', palette='viridis')
plt.title('Disponibilidad de Refrigeracion en Comunidades')
plt.xlabel('Refrigeracion Disponible')
plt.ylabel('Numero de Comunidades')

# 3. Diagrama de Dispersion - Distancia vs Demanda de Medicamentos
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='Distancia_a_Centro_Distribucion', y='Demanda_Medicamentos', hue='Nivel_Urgencia', palette='coolwarm')
plt.title('Demanda de Medicamentos vs Distancia al Centro de Distribucion')
plt.xlabel('Distancia al Centro de Distribucion (km)')
plt.ylabel('Demanda de Medicamentos')

# 4. Grafico de Barras - Frecuencia de Entrega por Nivel de Urgencia
plt.subplot(2, 2, 4)
sns.countplot(data=df, x='Frecuencia_Entrega', hue='Nivel_Urgencia', palette='muted')
plt.title('Frecuencia de Entrega por Nivel de Urgencia')
plt.xlabel('Frecuencia de Entrega')
plt.ylabel('Numero de Comunidades')

plt.tight_layout()
plt.show()
