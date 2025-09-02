import numpy as np

# Températures : lignes = villes (4), colonnes = jours (7)
temperature_data = np.array([
    [27, 20, 15, 18, 26, 18, 22],  # City 1
    [24, 18, 20, 17, 19, 22, 21],  # City 2
    [23, 23, 27, 25, 16, 21, 22],  # City 3
    [22, 29, 23, 16, 20, 24, 28]  # City 4
])

# 1. Moyenne par ville (ligne)
average_temp = np.mean(temperature_data, axis=1)

# 2. Max et Min par ville
max_temp = np.max(temperature_data, axis=1)
min_temp = np.min(temperature_data, axis=1)

# 3. Comparaison Jour 1 (col 0) vs Jour 7 (col 6)
day_comparison = temperature_data[:, 6] - temperature_data[:, 0]

# Affichage des résultats
print("=== Températures hebdomadaires ===\n")

for i in range(4):
    print(f"Ville {i + 1}:")
    print(f"  Moyenne : {average_temp[i]:.2f}°C")
    print(f"  Max     : {max_temp[i]}°C")
    print(f"  Min     : {min_temp[i]}°C")
    print(f"  Différence Jour1→Jour7 : {day_comparison[i]}°C")
    print()
