import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Descargar el dataset desde una URL directa
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
response = requests.get(url)
if response.status_code == 200:
    print("Dataset descargado exitosamente.")
    data = pd.read_csv(StringIO(response.text))
else:
    print(f"Error al descargar el dataset: {response.status_code}")
    exit()

# Preprocesamiento
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)  # Eliminar columnas irrelevantes
data['Age'] = data['Age'].fillna(data['Age'].mean())   # Imputar valores nulos en 'Age'
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # Imputar 'Embarked'

# Codificar variables categoricas
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Dividir en caracteristicas y objetivo
X = data.drop('Survived', axis=1)
y = data['Survived']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: Regresion Logistica
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Evaluar el modelo de Regresion Logistica
print("\nEvaluacion: Regresion Logistica")
print("Precision:", accuracy_score(y_test, log_pred))
print("Reporte de clasificacion:\n", classification_report(y_test, log_pred))

# Modelo 2: arbol de Decision
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

# Evaluar el modelo de arbol de Decision
print("\nEvaluacion: arbol de Decision")
print("Precision:", accuracy_score(y_test, tree_pred))
print("Reporte de clasificacion:\n", classification_report(y_test, tree_pred))
