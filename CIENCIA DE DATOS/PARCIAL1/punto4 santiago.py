import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from functools import partial

# Definición de constantes
NUM_CIUDADES = 10
TAMANO_POBLACION = 300
NUM_GENERACIONES = 400
PROB_CRUCE = 0.7
PROB_MUTACION = 0.2

# Generación de una matriz simétrica de distancias
def generar_matriz_distancias(n):
    matriz = np.random.randint(1, 100, size=(n, n))
    matriz = (matriz + matriz.T) / 2
    np.fill_diagonal(matriz, 0)
    return matriz

# Creación del problema de minimización
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)

# Creación de un individuo como una secuencia aleatoria de ciudades
def crear_individuo(ciudades):
    return random.sample(ciudades, len(ciudades))

# Evaluación del recorrido: suma de distancias entre ciudades consecutivas
def evaluar_tsp(ruta, distancias):
    total = sum(distancias[ruta[i]][ruta[i+1]] for i in range(len(ruta)-1))
    total += distancias[ruta[-1]][ruta[0]]
    return total,

# Configuración del algoritmo genético
toolbox = base.Toolbox()

# Ejecución del algoritmo
def ejecutar_algoritmo():
    distancias = generar_matriz_distancias(NUM_CIUDADES)
    coordenadas = np.random.rand(NUM_CIUDADES, 2) * 100
    
    # Definir el conjunto de ciudades y registrar la función con la lista de ciudades ya definida
    lista_ciudades = list(range(NUM_CIUDADES))
    toolbox.register("individuo", tools.initIterate, creator.Individuo, partial(crear_individuo, lista_ciudades))
    
    toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluar_tsp, distancias=distancias)
    
    poblacion = toolbox.poblacion(n=TAMANO_POBLACION)
    result = algorithms.eaSimple(poblacion, toolbox, cxpb=PROB_CRUCE, mutpb=PROB_MUTACION, ngen=NUM_GENERACIONES, verbose=False)
    
    mejor = tools.selBest(poblacion, k=1)[0]
    print(f"Mejor recorrido: {mejor}")
    print(f"Distancia mínima: {evaluar_tsp(mejor, distancias)[0]}")
    
    graficar_recorrido(mejor, coordenadas)

# Función para graficar el recorrido
def graficar_recorrido(mejor, coordenadas):
    plt.figure(figsize=(10, 6))
    for i, coord in enumerate(coordenadas):
        plt.scatter(*coord, color='blue', s=100)
        plt.text(coord[0]+1, coord[1], f'Ciudad {i}', fontsize=12, color='darkred', weight='bold')
    
    for i in range(len(mejor)-1):
        inicio, fin = mejor[i], mejor[i+1]
        plt.plot([coordenadas[inicio][0], coordenadas[fin][0]], [coordenadas[inicio][1], coordenadas[fin][1]], 'g-', linewidth=2)
    
    plt.plot([coordenadas[mejor[-1]][0], coordenadas[mejor[0]][0]], [coordenadas[mejor[-1]][1], coordenadas[mejor[0]][1]], 'g-', linewidth=2)
    plt.scatter(*coordenadas[mejor[0]], color='red', s=200, label='Inicio/Fin')
    plt.title("Recorrido óptimo", fontsize=14, weight='bold')
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True)
    plt.legend()
    plt.show()

# Ejecutar el algoritmo
ejecutar_algoritmo()
