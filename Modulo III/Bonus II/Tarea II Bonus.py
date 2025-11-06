
import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================
# 1. Configuración de Ciudades
# ==============================
# Se definen las ciudades con coordenadas
ciudades = {
    'MAZATLAN':    (5, 7),
    'TIJUANA':     (8, 4),
    'HERMOSILLO':  (2, 6),
    'MONTERREY':   (8, 1),
    'QUERETARO':   (6, 6),
    'GUADALAJARA': (8, 9),
    'COLIMA':      (6, 2),
}

# Se convierten los valores a listas para trabajarlas fácilmente
lista_ciudades = list(ciudades.values())
mapa_indices = list(ciudades.keys())
N_CIUDADES = len(lista_ciudades)

print(f"Problema del Vendedor Viajero con {N_CIUDADES} ciudades.\n")

# ==============================
# 2. Funciones auxiliares
# ==============================
def calcular_distancia(p1, p2):
    """Devuelve la distancia euclidiana entre dos puntos (x, y)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calcular_aptitud(ruta):
    """
    Calcula la distancia total recorrida en una ruta cerrada (vuelve al inicio).
    Mientras menor sea la distancia, mejor es la aptitud.
    """
    distancia_total = 0
    for i in range(N_CIUDADES - 1):
        distancia_total += calcular_distancia(lista_ciudades[ruta[i]], lista_ciudades[ruta[i+1]])
    distancia_total += calcular_distancia(lista_ciudades[ruta[-1]], lista_ciudades[ruta[0]])
    return distancia_total

def crear_poblacion(tamano_poblacion):
    """Genera rutas iniciales aleatorias (permutaciones de las ciudades)."""
    base = list(range(N_CIUDADES))
    return [random.sample(base, N_CIUDADES) for _ in range(tamano_poblacion)]

def seleccion_torneo(poblacion, aptitudes, tamano_torneo=3):
    """Selecciona la mejor ruta entre un grupo aleatorio (torneo)."""
    indices = random.sample(range(len(poblacion)), tamano_torneo)
    mejor = min(indices, key=lambda i: aptitudes[i])
    return poblacion[mejor]

def cruce_ordenado(padre1, padre2):
    hijo = [-1] * N_CIUDADES
    inicio, fin = sorted(random.sample(range(N_CIUDADES), 2))
    hijo[inicio:fin] = padre1[inicio:fin]
    pos = 0
    for i in range(N_CIUDADES):
        if hijo[i] == -1:
            while padre2[pos] in hijo:
                pos += 1
            hijo[i] = padre2[pos]
    return hijo

def mutacion_intercambio(ruta, tasa_mutacion):
    """Con cierta probabilidad, intercambia dos ciudades en la ruta."""
    if random.random() < tasa_mutacion:
        i, j = random.sample(range(N_CIUDADES), 2)
        ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta

# ==============================
# 3. Parámetros del Algoritmo Genético
# ==============================
TAMANO_POBLACION = 100   # Individuos por generación
GENERACIONES = 500       # Número de iteraciones
TASA_MUTACION = 0.05     # Probabilidad de mutar una ruta
TAMANO_TORNEO = 5        # Tamaño del grupo en la selección por torneo
PROB_CRUCE = 1.0         # Probabilidad de cruce (1 = siempre hay cruce)

print("Iniciando Algoritmo Genético...\n")

# ==============================
# 4. Proceso Evolutivo
# ==============================
poblacion = crear_poblacion(TAMANO_POBLACION)
mejor_ruta_global = None
mejor_aptitud_global = float('inf')

# Historiales para graficar la evolución
historial_mejor_gen = []
historial_promedio_gen = []
historial_mejor_global = []

for gen in range(GENERACIONES):
    aptitudes = [calcular_aptitud(r) for r in poblacion]
    mejor_aptitud_gen = min(aptitudes)
    promedio_aptitud_gen = np.mean(aptitudes)
    mejor_ruta_gen = poblacion[np.argmin(aptitudes)]
    # Actualizar el mejor resultado global
    if mejor_aptitud_gen < mejor_aptitud_global:
        mejor_aptitud_global = mejor_aptitud_gen
        mejor_ruta_global = mejor_ruta_gen

    # Guardar datos de convergencia
    historial_mejor_gen.append(mejor_aptitud_gen)
    historial_promedio_gen.append(promedio_aptitud_gen)
    historial_mejor_global.append(mejor_aptitud_global)

    # Crear nueva generación (elitismo + reproducción)
    nueva_poblacion = [mejor_ruta_global]  # Mantiene el mejor
    while len(nueva_poblacion) < TAMANO_POBLACION:
        padre1 = seleccion_torneo(poblacion, aptitudes, TAMANO_TORNEO)
        padre2 = seleccion_torneo(poblacion, aptitudes, TAMANO_TORNEO)
        if random.random() < PROB_CRUCE:
            hijo = cruce_ordenado(padre1, padre2)
        else:
            hijo = padre1[:]
        hijo = mutacion_intercambio(hijo, TASA_MUTACION)
        nueva_poblacion.append(hijo)
    poblacion = nueva_poblacion

    # Mostrar progreso cada 50 generaciones
    if (gen + 1) % 50 == 0:
        print(f"Generación {gen+1}/{GENERACIONES} -> Mejor distancia: {mejor_aptitud_global:.2f}")
# ==============================
# 5. Resultados Finales
# ==============================
ruta_optima_nombres = [mapa_indices[i] for i in mejor_ruta_global] + [mapa_indices[mejor_ruta_global[0]]]
print("\n--- RESULTADOS ---")
print(f"Mejor distancia total: {mejor_aptitud_global:.3f}")
print(f"Ruta óptima: {' -> '.join(ruta_optima_nombres)}")
# ==============================
# 6. Visualización de Resultados
# ==============================
plt.figure(figsize=(12, 6))
# --- Gráfica de la ruta óptima ---
plt.subplot(1, 2, 1)
coords = [lista_ciudades[i] for i in mejor_ruta_global] + [lista_ciudades[mejor_ruta_global[0]]]
x, y = zip(*coords)
plt.plot(x, y, 'ro-', linewidth=2.5, label='Ruta Óptima')
plt.scatter(x[0], y[0], color='lime', s=150, edgecolor='black', label='Inicio')
for i, nombre in enumerate(mapa_indices):
    plt.text(lista_ciudades[i][0]+0.1, lista_ciudades[i][1]+0.1, nombre, fontsize=11, weight='bold')
plt.title("Ruta Óptima del Viajero")
plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.legend()
plt.grid(True)

# --- Gráfica de convergencia ---
plt.subplot(1, 2, 2)
plt.plot(historial_mejor_gen, label="Mejor por generación", linewidth=2)
plt.plot(historial_promedio_gen, label="Promedio por generación", linewidth=2)
plt.plot(historial_mejor_global, label="Mejor global", linewidth=2)
plt.title("Convergencia del Algoritmo Genético")
plt.xlabel("Generación")
plt.ylabel("Distancia (Aptitud)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()