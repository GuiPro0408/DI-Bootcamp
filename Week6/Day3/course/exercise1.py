import numpy as np

array_1d = np.arange(1, 6)
array_2d = np.arange(1, 7).reshape(2, 3)

print(f"Array 1D: {array_1d}\nShape: {array_1d.shape}\nSize: {array_1d.size}\nData type: {array_1d.dtype}\n\n")
print(f"Array 2D: {array_2d}\nShape: {array_2d.shape}\nSize: {array_2d.size}\nData type: {array_2d.dtype}\n\n")

array_float = np.arange(1, 4, dtype=float)
print(f"Array Float: {array_float}\nShape: {array_float.shape}\nSize: {array_float.size}\nData type: {array_float.dtype}\n\n")
