import numpy as np

# Création de deux matrices 3x3
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
])

# Addition
add_result = A + B

# Soustraction
sub_result = A - B

# Multiplication élément par élément
elementwise_mul = A * B

# Multiplication matricielle (algèbre linéaire)
matrix_mul = np.dot(A, B)

# Résultats
print("Matrix A:\n", A)
print("\nMatrix B:\n", B)

print("\nAddition (A + B):\n", add_result)
print("\nSoustraction (A - B):\n", sub_result)
print("\nMultiplication élément par élément (A * B):\n", elementwise_mul)
print("\nMultiplication matricielle (A @ B):\n", matrix_mul)
