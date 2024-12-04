
import random
import math
from collections import defaultdict

# =======================
# 1. Algoritmo 2-SAT
# =======================
def add_implication(graph, u, v):
    graph[u].append(v)
    graph[v ^ 1].append(u ^ 1)

def dfs(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(graph, neighbor, visited, stack)
    stack.append(node)

def two_sat(n, clauses):
    graph = defaultdict(list)
    for u, v in clauses:
        add_implication(graph, u, v)
    visited = [False] * (2 * n)
    stack = []
    for i in range(2 * n):
        if not visited[i]:
            dfs(graph, i, visited, stack)
    visited = [False] * (2 * n)
    result = [False] * n
    while stack:
        node = stack.pop()
        if not visited[node ^ 1]:
            visited[node] = True
            visited[node ^ 1] = True
            result[node // 2] = (node % 2 == 0)
    return result

# =======================
# 2. Algoritmo TSP (Traveling Salesman Problem) usando 2-opt
# =======================
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def total_distance(tour, points):
    dist = 0
    for i in range(len(tour)):
        dist += distance(points[tour[i]], points[tour[(i+1) % len(tour)]])
    return dist

def two_opt(points):
    n = len(points)
    tour = list(range(n))
    best_dist = total_distance(tour, points)
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                new_tour = tour[:i+1] + list(reversed(tour[i+1:j+1])) + tour[j+1:]
                new_dist = total_distance(new_tour, points)
                if new_dist < best_dist:
                    tour = new_tour
                    best_dist = new_dist
                    improved = True
    return tour, best_dist

# =======================
# 3. Algoritmo de Escalado basado en funciones heuristicas (Recocido Simulado)
# =======================
def energy(x):
    return x**2 - 4*x + 4  # Ejemplo de una funcion cuadratica

def simulated_annealing(initial, temperature, cooling_rate, max_iter):
    current = initial
    best = current
    current_energy = energy(current)
    best_energy = current_energy

    for i in range(max_iter):
        # Proponer una nueva solucion
        new_solution = current + random.uniform(-1, 1)
        new_energy = energy(new_solution)
        
        # Decidir si aceptar la nueva solucion
        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / temperature):
            current = new_solution
            current_energy = new_energy

            if current_energy < best_energy:
                best = current
                best_energy = current_energy

        # Enfriar la temperatura
        temperature *= cooling_rate

    return best, best_energy

# =======================
# Main: Ejecutar los algoritmos
# =======================
if __name__ == "__main__":
    # =======================
    # 1. 2-SAT Ejemplo
    # =======================
    print("=== 2-SAT ===")
    n = 3  # Tres variables
    clauses = [(0, 1), (2, 3), (1, 2)]  # Clausulas (x1 or x2, x2 or x3, x1 or x3)
    result = two_sat(n, clauses)
    print("Resultado de 2-SAT:", result)

    # =======================
    # 2. TSP Ejemplo (con 2-opt)
    # =======================
    print("\n=== TSP ===")
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    tour, dist = two_opt(points)
    print("Ruta optimizada:", tour)
    print("Distancia total:", dist)

    # =======================
    # 3. Recocido Simulado Ejemplo
    # =======================
    print("\n=== Recocido Simulado ===")
    initial_solution = random.uniform(-10, 10)
    temperature = 1000
    cooling_rate = 0.99
    max_iter = 1000

    best_solution, best_energy = simulated_annealing(initial_solution, temperature, cooling_rate, max_iter)
    print("Mejor solucion encontrada:", best_solution)
    print("Energia (valor minimo):", best_energy)


