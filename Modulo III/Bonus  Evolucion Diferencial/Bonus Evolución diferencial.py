import numpy as np

# --- 1. Definir la Ecuación y Función de Costo ---
# Coeficientes de la ecuación cuadrática: A·x² + B·x + C
A = 15
B =8
C = 5

def funcion_costo(x):
    """
    Calcula el costo o error absoluto del individuo 'x' 
    al evaluarlo en la ecuación A·x² + B·x + C.
    El objetivo del algoritmo es minimizar este valor 
    (idealmente que el resultado sea 0).
    """
    y = (A * x**2) + (B * x) + C
    return abs(y)

# --- 2. Parámetros del Algoritmo Genético ---
TAM_POBLACION = 100    # Cantidad de individuos (soluciones candidatas)
GENERACIONES = 100     # Número total de iteraciones o generaciones
RANGO_BUSQ_MIN = -100.0  # Límite inferior del rango de búsqueda
RANGO_BUSQ_MAX = 100.0   # Límite superior del rango de búsqueda
RATIO_MUTAR = 0.1        # Probabilidad de que ocurra una mutación (10%)
FUERZA_MUTACION = 0.5    # Intensidad del cambio cuando ocurre mutación

# --- 3. Inicializar la Población ---
# Crear una población inicial con valores aleatorios entre -100 y 100
poblacion = np.random.uniform(RANGO_BUSQ_MIN, RANGO_BUSQ_MAX, TAM_POBLACION)

print(f"--- Resolviendo: {A}x² + {B}x + {C} = 0 ---")
print(f"Discriminante (b² - 4ac) = {B**2 - 4*A*C} (No hay soluciones reales)")
print("Ejecutando Algoritmo Genético para encontrar el valor 'x' que minimiza la función...")

# --- 4. Ciclo de Evolución ---
mejor_individual = None   # Guarda el mejor valor de 'x' encontrado
mejor_costo = np.inf       # Guarda el menor costo obtenido

for gen in range(GENERACIONES):
    
    # --- Evaluación (Fitness) ---
    # Calcula el costo de cada individuo de la población
    costos = np.array([funcion_costo(x) for x in poblacion])
    
    # Encuentra el mejor individuo de esta generación
    mejor_gen_idx = np.argmin(costos)
    mejor_gen_costo = costos[mejor_gen_idx]
    
    # Actualiza el mejor global si se encuentra uno mejor
    if mejor_gen_costo < mejor_costo:
        mejor_costo = mejor_gen_costo
        mejor_individual = poblacion[mejor_gen_idx]
        
    # --- Selección (Elitismo) ---
    # Ordena los individuos por su costo (de menor a mayor)
    elite_indices = np.argsort(costos)
    # Selecciona los 20 mejores como padres para la siguiente generación
    padres = poblacion[elite_indices[:20]]
    
    # --- Cruce (Crossover) y Mutación ---
    nueva_poblacion = []
    
    # Conserva los 20 mejores (elitismo)
    nueva_poblacion.extend(padres)
    
    # Genera el resto de la población (80 nuevos individuos)
    while len(nueva_poblacion) < TAM_POBLACION:
        # Selecciona dos padres al azar
        p1 = padres[np.random.randint(0, len(padres))]
        p2 = padres[np.random.randint(0, len(padres))]
        
        # Cruce: combinación simple de los padres (promedio)
        hijo = (p1 + p2) / 2.0
        
        # Aplica mutación con cierta probabilidad
        if np.random.rand() < RATIO_MUTAR:
            mutacion = np.random.uniform(-FUERZA_MUTACION, FUERZA_MUTACION)
            hijo += mutacion
            
        nueva_poblacion.append(hijo)
        
    # Actualiza la población con la nueva generación
    poblacion = np.array(nueva_poblacion)

# --- 5. Resultados ---
print(f"\nEvolución completada después de {GENERACIONES} generaciones.")
print(f"El Algoritmo Genético convergió a:")
print(f"  x = {mejor_individual:.6f}")
print(f"  Costo ( |{A}x² + {B}x + {C}| ) = {mejor_costo:.6f}")
